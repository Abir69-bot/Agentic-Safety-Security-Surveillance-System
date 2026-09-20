import { Component, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Bloom, EffectComposer } from '@react-three/postprocessing'

function NodeMarkers() {
  const group = useRef()
  const markers = useMemo(() => Array.from({ length: 12 }, (_, index) => {
    const latitude = -1.05 + (index % 4) * 0.7
    const longitude = index * 2.4
    const radius = 1.52
    return [radius * Math.cos(latitude) * Math.cos(longitude), radius * Math.sin(latitude), radius * Math.cos(latitude) * Math.sin(longitude)]
  }), [])
  useFrame(({ clock }) => { if (group.current) group.current.scale.setScalar(1 + Math.sin(clock.elapsedTime * 2.2) * 0.12) })
  return <group ref={group}>{markers.map((position, index) => <mesh key={index} position={position}><sphereGeometry args={[0.045, 8, 8]} /><meshStandardMaterial color="#22e5ff" emissive="#22e5ff" emissiveIntensity={4} toneMapped={false} /></mesh>)}</group>
}

function Globe() {
  const globe = useRef()
  useFrame((_, delta) => { if (globe.current) globe.current.rotation.y += delta * 0.12 })
  return <group ref={globe}><mesh><icosahedronGeometry args={[1.5, 2]} /><meshBasicMaterial color="#22e5ff" wireframe transparent opacity={0.34} /></mesh><mesh scale={0.88}><icosahedronGeometry args={[1.5, 2]} /><meshBasicMaterial color="#a06bff" wireframe transparent opacity={0.12} /></mesh><NodeMarkers /></group>
}

function Scene() { return <Canvas dpr={[1, 1.5]} camera={{ position: [0, 0, 4.5], fov: 42 }} gl={{ antialias: true, alpha: true }} fallback={<div className="orbit-fallback" aria-hidden="true"><span>PW</span></div>}><ambientLight intensity={0.35} /><Globe /><OrbitControls enableZoom={false} enablePan={false} enableDamping dampingFactor={0.08} minPolarAngle={Math.PI * 0.3} maxPolarAngle={Math.PI * 0.7} minAzimuthAngle={-Math.PI * 0.45} maxAzimuthAngle={Math.PI * 0.45} /><EffectComposer><Bloom intensity={1.25} luminanceThreshold={0.2} luminanceSmoothing={0.6} mipmapBlur /></EffectComposer></Canvas> }

class SceneBoundary extends Component {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <div className="orbit-fallback" aria-hidden="true"><span>PW</span><i /><i /><i /></div> : this.props.children }
}

export default function OrbitScene() {
  const [webglAvailable] = useState(() => { try { const canvas = document.createElement('canvas'); return Boolean(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))) } catch { return false } })
  return <div className="orbit-scene" aria-label="Interactive camera network visualization">{webglAvailable ? <SceneBoundary><Scene /></SceneBoundary> : <div className="orbit-fallback" aria-hidden="true"><span>PW</span><i /><i /><i /></div>}</div>
}
