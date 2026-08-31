// Minimal Three.js point-cloud viewer for the M1 scaffold.
//
// Goal (per the M1 spec): NOT visual polish. Just prove the frontend can
// load a scene produced by the real io -> depth -> reconstruction chain
// (scripts/export_sample_scene.py) and let the user orbit/pan/zoom it.
//
// NOT VERIFIED IN A BROWSER: this file was authored and syntax-reviewed in
// a headless development session with no browser and no network access to
// fetch the CDN-hosted Three.js module, so it could not actually be
// rendered or click-tested there. Open it in a real browser (see
// frontend/README.md) to verify.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const statusEl = document.getElementById("status");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  1000
);
camera.position.set(0, 0, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

async function loadScene(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

function buildPointCloud(sceneJson) {
  const n = sceneJson.points.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const [x, y, z, r, g, b] = sceneJson.points[i];
    positions[i * 3 + 0] = x;
    positions[i * 3 + 1] = -y; // flip so "up" in image-space looks up on screen
    positions[i * 3 + 2] = -z; // camera looks down -Z in Three.js by default
    colors[i * 3 + 0] = r / 255;
    colors[i * 3 + 1] = g / 255;
    colors[i * 3 + 2] = b / 255;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
  return new THREE.Points(geometry, material);
}

async function main() {
  try {
    const sceneJson = await loadScene("./sample_scene.json");
    const points = buildPointCloud(sceneJson);
    scene.add(points);

    const prov = sceneJson.provenance || {};
    statusEl.innerHTML = [
      `Loaded ${sceneJson.point_count.toLocaleString()} points.`,
      `<b>Depth source:</b> ${prov.depth_source} (${prov.depth_status})`,
      `<b>Calibrated:</b> ${sceneJson.calibrated}`,
      `<b style="color:#ff6b6b">${prov.warning || ""}</b>`,
    ].join("<br/>");
  } catch (err) {
    statusEl.innerHTML = `<b style="color:#ff6b6b">Failed to load scene:</b> ${err.message}`;
    console.error(err);
  }
}
main();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
