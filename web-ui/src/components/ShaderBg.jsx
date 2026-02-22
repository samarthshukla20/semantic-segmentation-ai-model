import { useEffect, useRef } from 'react';
import './ShaderBg.css';

const VERT = `
  attribute vec2 a_position;
  void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
`;

const FRAG = `
  precision highp float;
  uniform float u_time;
  uniform vec2  u_resolution;

  vec3 mod289(vec3 x) { return x - floor(x / 289.0) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x / 289.0) * 289.0; }
  vec3 permute(vec3 x) { return mod289((x * 34.0 + 1.0) * x); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x  = 2.0 * fract(p * C.www) - 1.0;
    vec3 h  = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float aspect = u_resolution.x / u_resolution.y;
    vec2 p = vec2(uv.x * aspect, uv.y);

    float t = u_time * 0.04;

    float n1 = snoise(p * 1.2 + vec2(t * 0.3, t * 0.2));
    float n2 = snoise(p * 2.5 + vec2(-t * 0.15, t * 0.4));
    float n3 = snoise(p * 0.6 + vec2(t * 0.1, -t * 0.08));
    float n  = n1 * 0.5 + n2 * 0.25 + n3 * 0.25;

    // Light desert palette
    vec3 col_a = vec3(0.98, 0.95, 0.90);   // warm white
    vec3 col_b = vec3(0.96, 0.91, 0.82);   // light sand
    vec3 col_c = vec3(0.92, 0.86, 0.75);   // mid sand
    vec3 col_d = vec3(0.88, 0.80, 0.66);   // golden sand
    vec3 col_e = vec3(0.82, 0.68, 0.42);   // amber accent

    float grad = smoothstep(0.0, 1.0, uv.y * 0.7 + 0.15);

    vec3 base = mix(col_a, col_b, smoothstep(-0.3, 0.3, n + grad * 0.5));
    base = mix(base, col_c, smoothstep(0.0, 0.6, n * 0.8 + grad));
    base = mix(base, col_d, smoothstep(0.3, 0.9, n3 * 0.5 + uv.y * 0.4) * 0.3);

    float highlight = smoothstep(0.35, 0.65, n1 * 0.6 + n3 * 0.4 + 0.3);
    base += col_e * highlight * 0.06;

    // soft vignette
    float vig = 1.0 - length((uv - 0.5) * vec2(1.0, 0.7)) * 0.35;
    base *= smoothstep(0.2, 0.8, vig);

    // subtle grain
    float grain = (fract(sin(dot(uv * u_time * 0.01, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * 0.012;
    base += grain;

    gl_FragColor = vec4(base, 1.0);
  }
`;

export default function ShaderBg() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', { alpha: false, antialias: false });
    if (!gl) return;

    function compile(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.warn('Shader compile error:', gl.getShaderInfoLog(s));
        gl.deleteShader(s);
        return null;
      }
      return s;
    }

    const vs = compile(gl.VERTEX_SHADER, VERT);
    const fs = compile(gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1,
    ]), gl.STATIC_DRAW);

    const aPos = gl.getAttribLocation(prog, 'a_position');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(prog, 'u_time');
    const uRes = gl.getUniformLocation(prog, 'u_resolution');

    let animId;
    let startTime = performance.now();

    function resize() {
      const dpr = Math.min(window.devicePixelRatio, 1.5);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      gl.viewport(0, 0, canvas.width, canvas.height);
    }
    resize();
    window.addEventListener('resize', resize);

    function render() {
      const elapsed = (performance.now() - startTime) / 1000;
      gl.uniform1f(uTime, elapsed);
      gl.uniform2f(uRes, canvas.width, canvas.height);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      animId = requestAnimationFrame(render);
    }
    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="shader-bg" />;
}
