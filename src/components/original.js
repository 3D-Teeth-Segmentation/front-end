import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { Container, Button, CircularProgress } from '@mui/material';
import { Link } from 'react-router-dom';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const SegmentationResults = () => {
    const mountRef = useRef(null);
    const [segmentedModel, setSegmentedModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Définir les couleurs pour les 15 classes
    const classColors = [
        [0.878, 0.878, 0.878], // Gray
        [0.839, 0.153, 0.157], // Red
        [0.121, 0.466, 0.705], // Blue
        [0.172, 0.627, 0.172], // Green
        [0.58, 0.404, 0.741], // Purple
        [1.0, 0.498, 0.054], // Orange
        [0.89, 0.467, 0.761], // Pink
        [0.498, 0.498, 0.498], // Gray
        [0.737, 0.741, 0.133], // Yellow
        [0.09, 0.745, 0.811], // Teal
        [0.682, 0.78, 0.909], // Light Blue
        [0.09, 0.745, 0.172], // Bright Green
        [0.831, 0.607, 0.101], // Gold
        [0.647, 0.38, 0.094], // Brown
        [0.596, 0.306, 0.639], // Dark Purple
        [0.18, 0.18, 0.18], // Dark Gray
    ];

    useEffect(() => {
        const cleanUpScene = () => {
            if (mountRef.current) {
                while (mountRef.current.firstChild) {
                    mountRef.current.removeChild(mountRef.current.firstChild);
                }
            }
        };

        const fileBase64 = localStorage.getItem('uploadedFile');
        if (!fileBase64) {
            setError("Aucun fichier n'a été téléchargé.");
            return;
        }

        const base64Data = fileBase64.split(',')[1];

        try {
            const byteString = atob(base64Data);
            const mimeString = fileBase64.split(',')[0].split(':')[1].split(';')[0];
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            const fileBlob = new Blob([ab], { type: mimeString });

            const segmentFile = async () => {
                setLoading(true);
                const formData = new FormData();
                formData.append('file', fileBlob, 'uploaded_file.obj');

                try {
                    console.log('Uploading');
                    const response = await fetch('http://127.0.0.1:8085/segmentation', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        throw new Error(`Erreur lors de la segmentation: ${response.statusText}`);
                    }

                    const data = await response.blob();
                    console.log('Données reçues:', data);
                    const url = URL.createObjectURL(data);
                    loadSegmentedModel(url);
                    setError(null);
                } catch (err) {
                    console.error('Erreur de requête:', err);
                    setError(err.message || 'Erreur de requête.');
                } finally {
                    setLoading(false);
                }
            };

            const applyClassColors = (object) => {
                let currentClassIndex = 0;
                object.traverse((child) => {
                    if (child.isMesh) {
                        const color = new THREE.Color(classColors[currentClassIndex % classColors.length]);
                        child.material = new THREE.MeshPhongMaterial({
                            color: color,
                            flatShading: true,
                        });
                        currentClassIndex++;
                    }
                });
            };

            const loadSegmentedModel = (meshPath) => {
                const loader = new OBJLoader();
                loader.load(
                    meshPath,
                    (object) => {
                        console.log('Modèle segmenté chargé:', object);
                        object.scale.set(0.01, 0.01, 0.01); // Ajuster l'échelle ici
                        setSegmentedModel(object);
                        renderModel(object);
                    },
                    (xhr) => {
                        console.log(`${(xhr.loaded / xhr.total) * 100}% chargé`);
                    },
                    (err) => {
                        console.error('Erreur lors du chargement du modèle:', err);
                        setError('Erreur lors du chargement du modèle.');
                    }
                );
            };

            const renderModel = (object) => {
                if (!mountRef.current) {
                    console.error('mountRef est null, impossible de rendre le modèle.');
                    return;
                }

                cleanUpScene();

                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(10, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer();
                renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
                mountRef.current.appendChild(renderer.domElement);

                const light = new THREE.DirectionalLight(0xffffff, 1);
                light.position.set(5, 5, 5).normalize();
                scene.add(light);
                scene.add(new THREE.AmbientLight(0x404040)); // Lumière ambiante

                object.position.set(0, 0, 0); // Centrer le modèle
                scene.add(object);

                camera.position.set(0, 0, 10); // Ajuster la position de la caméra
                camera.lookAt(0, 0, 0);

                const controls = new OrbitControls(camera, renderer.domElement);

                const animate = () => {
                    requestAnimationFrame(animate);
                    object.rotation.y += 0.01; // Rotation du modèle
                    renderer.render(scene, camera);
                    controls.update(); // Nécessaire si vous utilisez des contrôles
                };

                animate();
            };

            segmentFile();
        } catch (err) {
            console.error('Chaîne Base64 invalide:', err);
            setError('Chaîne Base64 invalide.');
        }

        return () => {
            cleanUpScene();
        };
    }, []);

    return (
        <Container>
            <h2>Résultats de Segmentation</h2>
            {loading ? (
                <CircularProgress />
            ) : error ? (
                <p style={{ color: 'red' }}>{error}</p>
            ) : (
                <div ref={mountRef} style={{ width: '800px', height: '500px', background: 'black', margin: '0 auto' }} />
            )}
            <div align="center" style={{ marginTop: 40, marginBottom: 20 }}>
                <Link to="/upload" style={{ textDecoration: 'none' }}>
                    <Button style={{ height: 50, width: 300 }} variant="contained" color="primary">
                        Nouvelle image
                    </Button>
                </Link>
            </div>
        </Container>
    );
};

export default SegmentationResults;
