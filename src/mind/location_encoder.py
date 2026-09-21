import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOCATION_ENCODER_METADATA_KEYS = (
    "capacity",
    "embed_dim",
    "harmonics_calculation",
    "legendre_polys",
    "le_type",
    "num_hidden_layers",
    "pe_type",
)
LOW_PRECISION_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
POLE_LATITUDE_EPS_DEGREES = 0.05


def _extract_location_encoder_metadata(hparams: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: hparams[key] for key in LOCATION_ENCODER_METADATA_KEYS}
    if metadata["le_type"] != "sphericalharmonics":
        raise NotImplementedError("expected le_type='sphericalharmonics'")
    if metadata["pe_type"] != "siren":
        raise NotImplementedError("expected pe_type='siren'")
    if metadata["harmonics_calculation"] != "analytic":
        raise NotImplementedError("standalone loader only supports analytic harmonics")
    return metadata


def _extract_location_encoder_state_dict(
    state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    location_state_dict = {
        key.split("nnet.", 1)[1]: value for key, value in state_dict.items() if "nnet." in key
    }
    if not location_state_dict:
        raise KeyError("could not find location encoder weights under an 'nnet.' prefix")
    return location_state_dict


def _load_location_encoder_payload(
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    standalone_formats = (
        "satclip-location-only",
        "satclip-standalone-location-encoder-state-dict",
    )
    if checkpoint.get("format") in standalone_formats:
        metadata = checkpoint["metadata"]
        if metadata["harmonics_calculation"] != "analytic":
            raise NotImplementedError("standalone loader only supports analytic harmonics")
        return metadata, checkpoint["state_dict"]
    return (
        _extract_location_encoder_metadata(checkpoint["hyper_parameters"]),
        _extract_location_encoder_state_dict(checkpoint["state_dict"]),
    )


def _resolve_posenc_compute_dtype(
    dtype: torch.dtype, posenc_compute_dtype: torch.dtype | None
) -> torch.dtype:
    if posenc_compute_dtype is not None:
        return posenc_compute_dtype
    if dtype in LOW_PRECISION_DTYPES:
        return torch.float32
    return dtype


def _normalize_nnet_state_dict(
    state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    if any(key.startswith("nnet.") for key in state_dict):
        return state_dict
    return {f"nnet.{key}": value for key, value in state_dict.items()}


class StableAnalyticSphericalHarmonics(nn.Module):
    orders: Tensor
    center_indices: Tensor
    diag_coeffs: Tensor
    subdiag_coeffs: Tensor
    alpha: Tensor
    beta: Tensor
    flat_degree: Tensor
    flat_m: Tensor
    negative_output_indices: Tensor
    positive_output_indices: Tensor

    def __init__(
        self,
        legendre_polys: int,
        compute_dtype: torch.dtype = torch.float64,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.L = int(legendre_polys)
        self.embedding_dim = self.L * self.L
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype
        self._init_constants()

    def _init_constants(self) -> None:
        orders = torch.arange(self.L, dtype=torch.int64)
        center_indices = orders * orders + orders

        diag_coeffs = torch.ones(self.L, dtype=torch.float64)
        if self.L > 1:
            diag_terms = torch.sqrt(1.0 + 1.0 / (2.0 * orders[1:].to(torch.float64)))
            diag_coeffs[1:] = torch.cumprod(diag_terms, dim=0)

        subdiag_coeffs = torch.sqrt(2.0 * orders[:-1].to(torch.float64) + 3.0)
        alpha = torch.zeros((self.L, self.L), dtype=torch.float64)
        beta = torch.zeros((self.L, self.L), dtype=torch.float64)
        flat_degree: list[int] = []
        flat_m: list[int] = []
        neg_index: list[int] = []
        pos_index: list[int] = []

        for degree in range(2, self.L):
            m = torch.arange(degree - 1, dtype=torch.float64)
            alpha[degree, : degree - 1] = torch.sqrt(
                ((2.0 * degree + 1.0) / (2.0 * degree - 3.0))
                * ((4.0 * (degree - 1) * (degree - 1) - 1.0) / (degree * degree - m * m))
            )
            beta[degree, : degree - 1] = torch.sqrt(
                ((2.0 * degree + 1.0) / (2.0 * degree - 3.0))
                * ((((degree - 1) * (degree - 1)) - m * m) / (degree * degree - m * m))
            )

        for degree in range(self.L):
            center_index = degree * degree + degree
            for m in range(1, degree + 1):
                flat_degree.append(degree)
                flat_m.append(m)
                neg_index.append(center_index - m)
                pos_index.append(center_index + m)

        self.register_buffer("orders", orders, persistent=False)
        self.register_buffer("center_indices", center_indices, persistent=False)
        self.register_buffer("diag_coeffs", diag_coeffs, persistent=False)
        self.register_buffer("subdiag_coeffs", subdiag_coeffs, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer(
            "flat_degree", torch.tensor(flat_degree, dtype=torch.int64), persistent=False
        )
        self.register_buffer("flat_m", torch.tensor(flat_m, dtype=torch.int64), persistent=False)
        self.register_buffer(
            "negative_output_indices",
            torch.tensor(neg_index, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "positive_output_indices",
            torch.tensor(pos_index, dtype=torch.int64),
            persistent=False,
        )

    def forward(self, lonlat: Tensor) -> Tensor:
        lonlat = lonlat.to(dtype=self.compute_dtype or lonlat.dtype)
        lon = lonlat[:, 0]
        lat = lonlat[:, 1].clamp(
            min=-90.0 + POLE_LATITUDE_EPS_DEGREES,
            max=90.0 - POLE_LATITUDE_EPS_DEGREES,
        )

        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)
        x = torch.cos(theta)
        u = torch.sqrt(torch.clamp(1 - x * x, min=0))

        dtype = lonlat.dtype
        device = lonlat.device
        batch_size = lonlat.shape[0]
        orders = self.orders.to(device=device)
        order_values = orders.to(dtype=dtype)

        p00 = lonlat.new_full((batch_size,), math.sqrt(1.0 / (4.0 * math.pi)))
        diag_terms: list[Tensor] = [p00]
        if self.L > 1:
            diag_coeffs = self.diag_coeffs.to(device=device, dtype=dtype)
            diag_values = (
                p00.unsqueeze(1)
                * diag_coeffs[1:].unsqueeze(0)
                * u.unsqueeze(1).pow(order_values[1:].unsqueeze(0))
            )
            diag_terms.extend(diag_values.unbind(dim=1))

        subdiag_coeffs = self.subdiag_coeffs.to(device=device, dtype=dtype)
        alpha = self.alpha.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)
        rows: list[Tensor] = [
            torch.cat(
                [diag_terms[0].unsqueeze(1), lonlat.new_zeros((batch_size, self.L - 1))],
                dim=1,
            )
        ]
        if self.L > 1:
            rows.append(
                torch.cat(
                    [
                        (subdiag_coeffs[0] * x * diag_terms[0]).unsqueeze(1),
                        diag_terms[1].unsqueeze(1),
                        lonlat.new_zeros((batch_size, self.L - 2)),
                    ],
                    dim=1,
                )
            )

        for degree in range(2, self.L):
            recurrence = (
                alpha[degree, : degree - 1].unsqueeze(0)
                * x.unsqueeze(1)
                * rows[degree - 1][:, : degree - 1]
                - beta[degree, : degree - 1].unsqueeze(0) * rows[degree - 2][:, : degree - 1]
            )
            row_terms: list[Tensor] = [
                recurrence,
                (subdiag_coeffs[degree - 1] * x * diag_terms[degree - 1]).unsqueeze(1),
                diag_terms[degree].unsqueeze(1),
            ]
            if degree < self.L - 1:
                row_terms.append(lonlat.new_zeros((batch_size, self.L - degree - 1)))
            rows.append(torch.cat(row_terms, dim=1))

        plm = torch.stack(rows, dim=1)
        phases = phi.unsqueeze(1) * order_values.unsqueeze(0)
        sin_terms = torch.sin(phases)
        cos_terms = torch.cos(phases)
        degree_blocks: list[Tensor] = []
        for degree in range(self.L):
            center = (math.pi * plm[:, degree, 0]).unsqueeze(1)
            if degree == 0:
                degree_blocks.append(center)
                continue
            m = orders[1 : degree + 1]
            base = math.sqrt(2.0) * plm[:, degree, 1 : degree + 1]
            negative = torch.flip(base * sin_terms[:, m], dims=(1,))
            positive = base * cos_terms[:, m]
            degree_blocks.append(torch.cat([negative, center, positive], dim=1))

        output = torch.cat(degree_blocks, dim=1)

        if self.output_dtype is not None:
            output = output.to(dtype=self.output_dtype)
        return output


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: nn.Module | None = None,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        scale = (1 / self.dim_in) if self.is_first else (math.sqrt(c / self.dim_in) / w0)
        weight.uniform_(-scale, scale)
        if bias is not None:
            bias.uniform_(-scale, scale)

        self.weight = nn.Parameter(weight)
        self.bias: nn.Parameter | None = nn.Parameter(bias) if bias is not None else None
        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        return self.activation(out)


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Siren(
                    dim_in=dim_in if index == 0 else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if index == 0 else w0,
                    use_bias=use_bias,
                    is_first=index == 0,
                    dropout=True,
                )
                for index in range(num_layers)
            ]
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=nn.Identity() if final_activation is None else final_activation,
            dropout=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class LocationEncoder(nn.Module):
    def __init__(
        self,
        posenc: StableAnalyticSphericalHarmonics,
        nnet: SirenNet,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.posenc = posenc
        self.nnet = nnet
        self.output_dtype = output_dtype

    @property
    def device(self) -> torch.device:
        return next(self.nnet.parameters()).device

    @property
    def nnet_dtype(self) -> torch.dtype:
        return next(self.nnet.parameters()).dtype

    def _forward_chunk(self, coords: Tensor) -> Tensor:
        coords = coords.to(device=self.device)
        encoded = self.posenc(coords)
        encoded = encoded.to(device=self.device, dtype=self.nnet_dtype)
        output = self.nnet(encoded)
        if self.output_dtype is not None and output.dtype != self.output_dtype:
            output = output.to(dtype=self.output_dtype)
        return output

    def forward(self, coords: Tensor, chunk_size: int | None = None) -> Tensor:
        if chunk_size is None:
            return self._forward_chunk(coords)
        return torch.cat([self._forward_chunk(chunk) for chunk in coords.split(chunk_size)], dim=0)


def build_pretrained_location_encoder(
    metadata: dict[str, Any],
    nnet_dtype: torch.dtype = torch.float64,
    posenc_compute_dtype: torch.dtype = torch.float64,
    output_dtype: torch.dtype | None = None,
) -> LocationEncoder:
    posenc = StableAnalyticSphericalHarmonics(
        legendre_polys=metadata["legendre_polys"],
        compute_dtype=posenc_compute_dtype,
        output_dtype=nnet_dtype,
    )
    nnet = SirenNet(
        dim_in=posenc.embedding_dim,
        dim_hidden=metadata["capacity"],
        dim_out=metadata["embed_dim"],
        num_layers=metadata["num_hidden_layers"],
    )
    return (
        LocationEncoder(posenc, nnet, output_dtype=output_dtype or nnet_dtype)
        .to(dtype=nnet_dtype)
        .eval()
    )


def load_pretrained_location_encoder(
    checkpoint_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    posenc_compute_dtype: torch.dtype | None = None,
) -> LocationEncoder:
    metadata, state_dict = _load_location_encoder_payload(checkpoint_path)
    model = build_pretrained_location_encoder(
        metadata,
        nnet_dtype=dtype,
        posenc_compute_dtype=_resolve_posenc_compute_dtype(dtype, posenc_compute_dtype),
        output_dtype=dtype,
    )
    model.load_state_dict(_normalize_nnet_state_dict(state_dict))
    return model.to(device=device).eval()
