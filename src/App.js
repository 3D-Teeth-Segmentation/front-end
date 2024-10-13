import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './components/Home';
import Footer from './components/Footer';
import UploadViewComponent from './components/UploadViewComponent';
import Contact from './components/Contact';
import AboutUs from './components/AboutUs'; // Import du composant AboutUs
import SegmentationResults from './components/SegmentationResults';

function App() {
    // const [model, setModel] = useState(null);  // Pour gérer le modèle 3D
    const [themeMode, setThemeMode] = useState('light');  // Pour le thème

    // Gestion du thème avec localStorage
    useEffect(() => {
        // Lire le thème sauvegardé ou utiliser un thème par défaut
        const savedTheme = localStorage.getItem('theme') || 'light';
        setThemeMode(savedTheme);
        document.body.className = savedTheme;

        // Pas besoin de nettoyage ici
    }, []);

    const changeThemeMode = (mode) => {
        setThemeMode(mode);
        localStorage.setItem('theme', mode);
        document.body.className = mode;
    };

    // Style de l'application basé sur le thème
    const appStyles = {
        backgroundColor: themeMode === 'dark' ? '#333' : '#fff',
        color: themeMode === 'dark' ? '#fff' : '#000',
        minHeight: '100vh',
    };

    // Fonction appelée lorsque le modèle est chargé
    // const handleModelLoaded = (loadedModel) => {
    //     setModel(loadedModel);
    // };

    return (
        <Router>
            <div style={appStyles}>
                {/* Navbar avec gestion du thème */}
                <Navbar themeMode={themeMode} changeThemeMode={changeThemeMode} />
                <br></br>
                {/* Routes pour la navigation */}
                <Routes>
                    <Route path="/contact" element={<Contact />} />

                    <Route path="/about" element={<AboutUs />} /> {/* Nouvelle route About Us */}

                    <Route path="/" element={<Home />} />

                    <Route path="/upload" element={<UploadViewComponent />} />

                    <Route path="/segmentation" element={<SegmentationResults />} />

                </Routes>
                <br></br>
                <Footer />
            </div>
        </Router>
    );
}

export default App;