import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import JSZip from 'jszip'; // Importer JSZip
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader';
import { Container, Button, CircularProgress } from '@mui/material';
import { Link } from 'react-router-dom';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const SegmentationResults = () => {
    const mountRef = useRef(null);
    const [segmentedModel, setSegmentedModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

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
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            const fileBlob = new Blob([ab], { type: 'application/octet-stream' });

            const segmentFile = async () => {
                setLoading(true);
                const formData = new FormData();
                formData.append('file', fileBlob, 'uploaded_file.obj');

                try {
                    const response = await fetch('http://127.0.0.1:8085/segmentation', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        throw new Error(`Erreur lors de la segmentation: ${response.statusText}`);
                    }

                    const zipBlob = await response.blob();
                    const zip = await JSZip.loadAsync(zipBlob);

                    console.log('Contenu du fichier ZIP:', zip.files);

                    const objFile = zip.file("Sample__predicted.obj");
                    const mtlFile = zip.file("Sample__predicted.mtl");

                    if (!objFile || !mtlFile) {
                        throw new Error("Les fichiers .obj ou .mtl ne sont pas trouvés dans le fichier ZIP.");
                    }

                    console.log('Fichier OBJ trouvé:', objFile);
                    console.log('Fichier MTL trouvé:', mtlFile);

                    // Lire les fichiers .obj et .mtl
                    const objData = await objFile.async("string");
                    const mtlData = await mtlFile.async("string");

                    // Convertir les données .obj et .mtl en Blob
                    const objBlob = new Blob([objData], { type: 'text/plain' });
                    const mtlBlob = new Blob([mtlData], { type: 'text/plain' });

                    // Créer des URLs pour les fichiers .obj et .mtl
                    const objUrl = URL.createObjectURL(objBlob);
                    const mtlUrl = URL.createObjectURL(mtlBlob);

                    console.log('URL générée pour OBJ:', objUrl);
                    console.log('URL générée pour MTL:', mtlUrl);

                    // Charger et afficher le modèle segmenté avec les URLs
                    loadSegmentedModel(objUrl, mtlUrl);

                    setError(null);
                } catch (err) {
                    console.error('Erreur de requête:', err);
                    setError(err.message || 'Erreur de requête.');
                } finally {
                    setLoading(false);
                }
            };

            const loadSegmentedModel = (objPath, mtlPath) => {
                const mtlLoader = new MTLLoader();

                console.log('Chargement du fichier MTL depuis:', mtlPath);

                mtlLoader.load(mtlPath, (materials) => {
                    console.log('Matériaux chargés:', materials);
                    materials.preload();

                    const objLoader = new OBJLoader();
                    objLoader.setMaterials(materials); // Appliquer les matériaux
                    objLoader.load(
                        objPath,
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
                            console.error('Erreur lors du chargement du modèle OBJ:', err);
                            setError('Erreur lors du chargement du modèle OBJ.');
                        }
                    );
                });
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

                // Lumière directionnelle
                const light = new THREE.DirectionalLight(0xffffff, 2); // Augmente l'intensité
                light.position.set(5, 5, 5).normalize();
                scene.add(light);

                // Ajouter une seconde lumière directionnelle pour un meilleur éclairage
                const light2 = new THREE.DirectionalLight(0xffffff, 2); // Lumière supplémentaire
                light2.position.set(-5, 5, 5).normalize(); // Positionner l'autre lumière
                scene.add(light2);

                // Lumière ambiante plus forte
                scene.add(new THREE.AmbientLight(0xffffff, 1)); // Augmente l'intensité de la lumière ambiante

                // Optionnel : Ajouter une lumière ponctuelle pour éclairer davantage l'objet
                const additionalLight = new THREE.PointLight(0xffffff, 1, 100); // Lumière ponctuelle
                additionalLight.position.set(0, 10, 0); // Positionner la lumière
                scene.add(additionalLight);

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
                <div ref={mountRef} style={{ width: '800px', height: '500px', background: 'whitesmoke', margin: '0 auto' }} />
            )}
            <div align="center" style={{ marginTop: '20px' }}>
                <Button component={Link} to="/upload" variant="contained" color="primary">
                    New File
                </Button>
            </div>
        </Container>
    );
};

export default SegmentationResults;
