import React, { useState } from 'react';
import { Button, TextField } from '@mui/material';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';

function UploadForm({ onModelLoaded, onFileUploaded }) {
    const [file, setFile] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (file) {
            console.log('Fichier téléchargé :', file);

            // Lire le fichier et le charger avec OBJLoader
            const reader = new FileReader();
            reader.onload = (e) => {
                const objLoader = new OBJLoader();
                const text = e.target.result;
                const object = objLoader.parse(text);
                console.log('Objet chargé :', object);

                // Passer l'objet chargé au composant parent
                onModelLoaded(object);
            };
            reader.readAsText(file);

            // Appelle la fonction pour informer le composant parent que le fichier a été téléchargé
            onFileUploaded(file);
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <TextField
                type="file"
                onChange={handleFileChange}
                inputProps={{ accept: '.obj,.json' }}
                style={{ flex: 1, backgroundColor: "white" }}
            />
            <Button sx={{ height: 55, width: 150, fontSize: 17 }} type="submit" variant="contained" color="primary">
                Upload
            </Button>
        </form>
    );
}

export default UploadForm;
