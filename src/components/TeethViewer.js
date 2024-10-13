import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import RotateRightIcon from '@mui/icons-material/RotateRight';

function TeethViewer({ model }) {
    const mountRef = useRef(null);

    useEffect(() => {
        // Création de la scène, de la caméra et du rendu
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();

        // Taille du rendu
        const renderWidth = window.innerWidth * 0.6;
        const renderHeight = window.innerHeight * 0.6;
        renderer.setSize(renderWidth, renderHeight);
        renderer.setClearColor(0x000000); // Arrière-plan noir

        // Append canvas
        if (mountRef.current) {
            mountRef.current.appendChild(renderer.domElement);
        }

        // Lumières
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5).normalize();
        scene.add(directionalLight);

        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);

        if (model) {
            // Taille et position du modèle
            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());
            model.position.sub(center);
            scene.add(model);
            camera.position.z = Math.max(size.x, size.y, size.z) * 2;
        }

        // Contrôles pour rotation
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.enableZoom = true;

        // Fonction d'animation
        const animate = function () {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };

        animate();

        // Nettoyage à la suppression du composant
        return () => {
            controls.dispose();  // Nettoyage des contrôles
            if (mountRef.current && renderer.domElement) {
                mountRef.current.removeChild(renderer.domElement); // Vérification stricte
            }
        };
    }, [model]);

    return (
        <div style={{ position: 'relative', width: '60vw', height: '60vh', margin: '0 auto', marginTop: '20px', border: '1px solid #ccc' }}>
            <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

            {/* Icône de rotation */}
            <div style={{ position: 'absolute', top: '10px', right: '10px', display: 'flex', alignItems: 'center', background: 'rgba(255, 255, 255, 0.7)', padding: '5px', borderRadius: '5px' }}>
                <RotateRightIcon style={{ marginRight: '5px' }} />
                <span style={{ fontSize: '12px' }}>Rotate the model</span>
            </div>
        </div>
    );
}

export default TeethViewer;