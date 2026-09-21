/* Orthographic globe with precomputed PCA textures; no map library or tile requests. */
(function () {
  "use strict";

  window.createMINDGlobe = function (canvas) {
    var gl = canvas.getContext("webgl", { alpha: true, antialias: false, depth: false });
    if (!gl) throw new Error("WebGL unavailable");
    var vertex = `
      attribute vec2 position;
      varying vec2 point;
      void main() { point = position * 1.12; gl_Position = vec4(position, 0.0, 1.0); }
    `;
    var fragment = `
      precision highp float;
      varying vec2 point;
      uniform sampler2D before, after, terrain;
      uniform float terrainReady;
      uniform float beforeFull, afterFull, blend, longitude, latitude, resolution;
      const float PI = 3.141592653589793;
      const vec3 OCEAN = vec3(0.14, 0.16, 0.17);

      vec2 equalEarth(float lon, float lat) {
        // WGS84 authalic latitude, followed by the Equal Earth forward projection.
        float e = 0.08181919084;
        float s = sin(lat);
        float q = (1.0-e*e) * (s/(1.0-e*e*s*s) - log((1.0-e*s)/(1.0+e*s))/(2.0*e));
        float theta = asin(0.86602540378 * clamp(q/1.9955310875, -1.0, 1.0));
        float t2 = theta*theta, t6 = t2*t2*t2;
        float x = lon*cos(theta)/(0.86602540378*(1.340264 - 0.243318*t2 + t6*(0.006251 + 0.034164*t2)));
        float y = theta*(1.340264 - 0.081106*t2 + t6*(0.000893 + 0.003796*t2));
        // Plot edges (14,14)–(1586,780) in the existing 1600×794 Equal Earth figure.
        return vec2((14.0 + (x/5.4132599674 + 0.5)*1572.0)/1600.0,
                    (14.0 + (0.5 - y/2.634725519)*766.0)/794.0);
      }

      vec3 colorAt(sampler2D map, float full, float lon, float lat) {
        // Storage thumbnails use the exact cell edges recorded in store.json.
        vec2 uv = vec2((degrees(lon)+180.0)/359.68, (90.0-degrees(lat))/179.84);
        if (full > 0.5) uv = equalEarth(lon, lat);
        vec4 color = texture2D(map, clamp(uv, 0.0, 1.0));
        if (full > 0.5) {
          // The full-field figure has an opaque #eef1f4 ocean; chunks have alpha.
          color.a = smoothstep(0.015, 0.06, distance(color.rgb, vec3(238.0,241.0,244.0)/255.0));
          // Ignore the figure's white antialiased border at the dateline and poles.
          if (abs(lon) > 3.13 || abs(lat) > 1.5) {
            color.a *= 1.0-smoothstep(0.88, 0.92, min(color.r, min(color.g, color.b)));
          }
        }
        return mix(OCEAN, color.rgb, color.a);
      }

      float heightAt(vec2 uv) {
        return texture2D(terrain, vec2(fract(uv.x), clamp(uv.y, 0.0, 1.0))).r;
      }

      vec3 reliefNormal(vec3 normal, float lon, float lat) {
        if (terrainReady < 0.5) return normal;
        vec2 uv = vec2(lon/(2.0*PI)+0.5, 0.5-lat/PI);
        vec2 step = vec2(1.0/2048.0, 1.0/1024.0);
        // Height is 0–9000 m; exaggerate slopes 16× to read at this small scale.
        float scale = 16.0*9000.0/6371000.0;
        scale *= 1.0-smoothstep(radians(80.0), radians(84.5), abs(lat));
        float eastSlope = (heightAt(uv+vec2(step.x,0.0))-heightAt(uv-vec2(step.x,0.0)))
                          *scale/(4.0*PI*step.x*max(cos(lat),0.1));
        float northSlope = (heightAt(uv-vec2(0.0,step.y))-heightAt(uv+vec2(0.0,step.y)))
                           *scale/(2.0*PI*step.y);
        float angle = lon-longitude;
        vec3 east = vec3(cos(angle), sin(angle)*sin(latitude), -sin(angle)*cos(latitude));
        vec3 north = vec3(-sin(lat)*sin(angle),
                         cos(lat)*cos(latitude)+sin(lat)*cos(angle)*sin(latitude),
                         cos(lat)*sin(latitude)-sin(lat)*cos(angle)*cos(latitude));
        return normalize(normal-eastSlope*east-northSlope*north);
      }

      void main() {
        float radius = length(point);
        if (radius > 1.0) { gl_FragColor = vec4(0.0); return; }
        vec3 normal = vec3(point, sqrt(max(0.0, 1.0-radius*radius)));
        float y = normal.y*cos(latitude) + normal.z*sin(latitude);
        float z = normal.z*cos(latitude) - normal.y*sin(latitude);
        float lat = asin(clamp(y, -1.0, 1.0));
        float lon = mod(atan(normal.x,z)+longitude+PI, 2.0*PI)-PI;
        vec3 color = mix(colorAt(before,beforeFull,lon,lat), colorAt(after,afterFull,lon,lat), blend);
        vec3 surface = reliefNormal(normal, lon, lat);
        float light = 0.64 + 0.36*max(0.0, dot(surface, normalize(vec3(-0.5,0.7,1.5))));
        color *= light * (0.72 + 0.28*pow(normal.z, 0.3));
        float rim = pow(1.0-normal.z, 4.0)*0.18;
        color = mix(color, vec3(0.50,0.63,0.85), rim);
        float alpha = 0.90*(1.0-smoothstep(1.0-2.24/resolution, 1.0, radius));
        gl_FragColor = vec4(color*alpha, alpha);
      }
    `;

    function shader(type, source) {
      var result = gl.createShader(type);
      gl.shaderSource(result, source);
      gl.compileShader(result);
      if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
        var message = gl.getShaderInfoLog(result);
        gl.deleteShader(result);
        throw new Error(message);
      }
      return result;
    }
    var program = gl.createProgram();
    var vs = shader(gl.VERTEX_SHADER, vertex), fs = shader(gl.FRAGMENT_SHADER, fragment);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("Globe shader failed");
    gl.useProgram(program);
    var buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    var position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    var uniforms = {};
    ["before", "after", "terrain", "terrainReady", "beforeFull", "afterFull", "blend", "longitude", "latitude", "resolution"].forEach(function (name) {
      uniforms[name] = gl.getUniformLocation(program, name);
    });
    gl.uniform1i(uniforms.before, 0);
    gl.uniform1i(uniforms.after, 1);
    gl.uniform1i(uniforms.terrain, 2);
    var terrain = null;

    return {
      setTerrain: function (image) { terrain = this.texture(image); },
      texture: function (image) {
        var texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        if (gl.getError() !== gl.NO_ERROR) throw new Error("Globe texture unavailable");
        return texture;
      },
      draw: function (previous, current, mix, lon, lat) {
        var size = Math.max(1, Math.round(canvas.clientWidth * Math.min(window.devicePixelRatio || 1, 2)));
        if (canvas.width !== size) canvas.width = canvas.height = size;
        gl.viewport(0, 0, size, size);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, previous.texture);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, current.texture);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, terrain || previous.texture);
        gl.uniform1f(uniforms.terrainReady, terrain ? 1 : 0);
        gl.uniform1f(uniforms.beforeFull, previous.full ? 1 : 0);
        gl.uniform1f(uniforms.afterFull, current.full ? 1 : 0);
        gl.uniform1f(uniforms.blend, mix);
        gl.uniform1f(uniforms.longitude, lon);
        gl.uniform1f(uniforms.latitude, lat);
        gl.uniform1f(uniforms.resolution, size);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    };
  };
})();
