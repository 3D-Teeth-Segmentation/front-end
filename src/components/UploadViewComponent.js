import React, { useState } from 'react';
import UploadForm from './UploadForm';
import TeethViewer from './TeethViewer';
import { Container, Button } from '@mui/material';
import { Link } from 'react-router-dom';

function UploadViewComponent() {
    const [model, setModel] = useState(null);
    const [uploadedFile, setUploadedFile] = useState(null);

    const handleModelLoaded = (loadedModel) => {
        setModel(loadedModel);
    };

    const handleFileUploaded = (file) => {
        setUploadedFile(file);

        // Convertir le fichier en Base64 avant de le stocker
        const reader = new FileReader();
        reader.onloadend = () => {
            localStorage.setItem('uploadedFile', reader.result); // Stocke le fichier Base64 dans localStorage
        };
        reader.readAsDataURL(file); // Lire le fichier comme une URL de données (Base64)
    };

    return (
        <div>
            <Container>
                <h1>3D Teeth Segmentation</h1>
                <UploadForm onModelLoaded={handleModelLoaded} onFileUploaded={handleFileUploaded} />

                <div style={{ marginBottom: '32px' }}>
                    {model && <h2>Model loaded: {model.name}</h2>}
                    <TeethViewer model={model} />
                </div>

                <div align={'center'}>
                    {uploadedFile && (
                        <Link to="/segmentation" style={{ textDecoration: 'none' }}>
                            <Button style={{ height: 50, width: 300 }} variant="contained" color="primary">
                                Apply Segmentation
                            </Button>
                        </Link>
                    )}
                </div>
            </Container>
        </div>
    );
}

export default UploadViewComponent;
