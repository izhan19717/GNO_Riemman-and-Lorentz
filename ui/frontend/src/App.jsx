import React, { useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Text } from '@react-three/drei';
import * as THREE from 'three';
import axios from 'axios';

function PointCloud({ data, manifold }) {
  const points = useMemo(() => {
    if (!data) return null;
    const vertices = [];
    const colors = [];
    const colorScale = (val) => {
      // Simple heatmap: blue -> red
      // Normalize val roughly [-1, 1]
      const t = (val + 1) / 2;
      return new THREE.Color().setHSL(0.7 * (1 - t), 1, 0.5);
    };

    data.forEach(p => {
      if (manifold === 'sphere') {
        vertices.push(p.x, p.y, p.z);
      } else {
        // Minkowski: t -> y, x -> x, val -> color
        // Map t to Y axis, x to X axis
        vertices.push(p.x - 3.14, p.t - 0.5, 0);
      }
      const c = colorScale(p.val);
      colors.push(c.r, c.g, c.b);
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return geometry;
  }, [data, manifold]);

  if (!points) return null;

  return (
    <points>
      <bufferGeometry attach="geometry" {...points} />
      <pointsMaterial attach="material" vertexColors size={0.1} sizeAttenuation={true} />
    </points>
  );
}

function App() {
  const [manifold, setManifold] = useState('sphere');
  const [tab, setTab] = useState('inference'); // 'inference' or 'benchmark'
  const [info, setInfo] = useState(null);
  const [inferenceData, setInferenceData] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(null);

  useEffect(() => {
    // Get Info
    axios.post('http://localhost:8000/manifold/info', { type: manifold })
      .then(res => setInfo(res.data))
      .catch(err => console.error(err));

    // Run Inference
    if (tab === 'inference') {
      axios.post('http://localhost:8000/inference', { type: manifold })
        .then(res => {
          if (res.data.status === 'success') {
            setInferenceData(res.data.data);
          }
        })
        .catch(err => console.error(err));
    } else {
      // Fetch Benchmark
      axios.get('http://localhost:8000/benchmark/results')
        .then(res => setBenchmarkData(res.data.data))
        .catch(err => console.error(err));
    }
  }, [manifold, tab]);

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#111', color: 'white', fontFamily: 'sans-serif' }}>
      <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 10 }}>
        <h1>Geometric Neural Operator</h1>
        <div style={{ marginBottom: 20 }}>
          <button onClick={() => setManifold('sphere')} style={{ marginRight: 10, padding: '10px 20px', background: manifold === 'sphere' ? '#4CAF50' : '#333', color: 'white', border: 'none', cursor: 'pointer' }}>
            Sphere
          </button>
          <button onClick={() => setManifold('minkowski')} style={{ padding: '10px 20px', background: manifold === 'minkowski' ? '#2196F3' : '#333', color: 'white', border: 'none', cursor: 'pointer' }}>
            Minkowski
          </button>
        </div>

        <div style={{ marginBottom: 20 }}>
          <button onClick={() => setTab('inference')} style={{ marginRight: 10, padding: '5px 10px', background: tab === 'inference' ? '#666' : '#222', color: 'white', border: '1px solid #444', cursor: 'pointer' }}>
            Inference
          </button>
          <button onClick={() => setTab('benchmark')} style={{ padding: '5px 10px', background: tab === 'benchmark' ? '#666' : '#222', color: 'white', border: '1px solid #444', cursor: 'pointer' }}>
            Scientific Benchmark
          </button>
        </div>

        {tab === 'inference' && info && (
          <div style={{ background: 'rgba(0,0,0,0.5)', padding: 20, borderRadius: 8 }}>
            <h3>{info.type.toUpperCase()}</h3>
            <p>{info.description}</p>
            <p>Dim: {info.dim}</p>
            <div style={{ marginTop: 20 }}>
              <button onClick={() => {
                setInferenceData(null);
                axios.post('http://localhost:8000/inference', { type: manifold })
                  .then(res => {
                    if (res.data.status === 'success') {
                      setInferenceData(res.data.data);
                    }
                  });
              }} style={{ padding: '10px 20px', background: '#FF9800', color: 'white', border: 'none', cursor: 'pointer' }}>
                Run New Inference
              </button>
            </div>
          </div>
        )}

        {tab === 'benchmark' && (
          <div style={{ background: 'rgba(0,0,0,0.8)', padding: 20, borderRadius: 8, maxWidth: '400px' }}>
            <h3>Convergence Study</h3>
            <p>Relative L2 Error vs Resolution</p>
            <img src="http://localhost:8000/benchmark/plot" alt="Convergence Plot" style={{ width: '100%', borderRadius: 4 }} />
            {benchmarkData && (
              <pre style={{ fontSize: '12px', marginTop: 10 }}>
                {JSON.stringify(benchmarkData, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>

      {tab === 'inference' && (
        <Canvas camera={{ position: [0, 0, 5] }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />

          {manifold === 'sphere' && (
            <>
              <Sphere args={[0.95, 32, 32]}>
                <meshStandardMaterial color="#333" wireframe />
              </Sphere>
              <PointCloud data={inferenceData} manifold="sphere" />
            </>
          )}

          {manifold === 'minkowski' && (
            <>
              <group>
                <gridHelper args={[10, 10]} rotation={[1.57, 0, 0]} position={[0, 0, -0.1]} />
                <Text position={[0, -2, 0]} fontSize={0.5} color="white">Space (x)</Text>
                <Text position={[-4, 0, 0]} fontSize={0.5} rotation={[0, 0, 1.57]} color="white">Time (t)</Text>
                <PointCloud data={inferenceData} manifold="minkowski" />
              </group>
            </>
          )}

          <OrbitControls />
        </Canvas>
      )}
    </div>
  );
}

export default App;
