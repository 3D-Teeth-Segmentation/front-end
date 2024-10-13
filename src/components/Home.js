import React, { useState, useEffect } from 'react';
import logo from '../images/logo.png';
import img_home from '../images/img_home.webp';
import { Link } from 'react-router-dom';


const Home = () => {
    const [isHovered, setIsHovered] = useState(false);

    // Fonction pour faire défiler jusqu'à la section du panel bleu
    const scrollToPanel = () => {
        const panelSection = document.getElementById('blue-panel');
        panelSection.scrollIntoView({ behavior: 'smooth' });
    };

    // Les styles pour chaque section
    const styles = {
        whyTeethSeg: {
            backgroundColor: '#e0eaf4',
            textAlign: 'center',
            padding: '50px 20px',
        },
        whyTeethSegTitle: {
            color: '#29b6f6',
            fontSize: '36px',
            marginBottom: '20px',
        },
        whyTeethSegText: {
            fontSize: '18px',
            color: '#555',
            maxWidth: '800px',
            margin: '0 auto 40px',
        },
        teethSeg: {
            backgroundColor: 'black',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingTop: '50px',
            paddingBottom: '50px',
            paddingLeft: '150px',
            paddingRight: '150px',
            height: 450,
            marginTop: 40
        },
        content: {
            maxWidth: '600px',
        },
        title: {
            fontSize: '40px',
            color: '#007aff',
        },
        description: {
            fontSize: '18px',
            margin: '20px 0',
            color: 'white'
        },
        button: {
            padding: '10px 20px',
            fontSize: '16px',
            fontWeight: 'bold',
            borderRadius: '5px',
            backgroundColor: '#007aff', // Couleur de fond
            color: '#fff',              // Couleur du texte
            border: 'none',
            cursor: 'pointer',
            transition: 'background-color 0.3s ease', // Transition pour un hover smooth
        },
        buttonHover: {
            backgroundColor: '#005bb5', // Couleur sur hover
        },
        imageContainer: {
            textAlign: 'right',
        },
        image: {
            width: '400px',
        },
        panel: {
            backgroundColor: '#0078d7', // Bleu
            padding: '50px',
            textAlign: 'center',
            marginTop: 180,
            marginBottom: 160
        },
        textContainer: {
            display: 'inline-block',
            position: 'relative',
            color: 'white',
            fontSize: '2.5em',
            fontWeight: 'bold',
        },
        animatedText: {
            display: 'inline-block',
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            animation: 'typing 5s steps(50, end), blink-caret 0.75s step-end infinite',
            borderRight: '3px solid white', // Pour simuler le curseur
        },
        demoButtonContainer: {
            marginTop: '20px',
            textAlign: 'center', // Centrer le bouton horizontalement
        },
        demoButton: {
            backgroundColor: 'white',
            color: '#0078d7',
            padding: '10px 20px',
            fontSize: '1em',
            fontWeight: 'bold',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
        },
        demoButtonHover: {
            backgroundColor: '#e0e0e0',
        },
    };

    useEffect(() => {
        const globalStyles = `
            @keyframes typing {
                from { width: 0; }
                to { width: 100%; }
            }
            @keyframes blink-caret {
                from, to { border-color: transparent; }
                50% { border-color: white; }
            }
        `;
        const styleElement = document.createElement('style');
        styleElement.innerHTML = globalStyles;
        document.head.appendChild(styleElement);
    }, []); // Exécution lors du montage du composant

    return (
        <div>
            {/* Section 1: TeethSeg */}
            <section style={styles.teethSeg}>
                <div style={styles.content}>
                    <h1 style={styles.title}>
                        <img src={logo} alt="Logo" style={{ height: '40px', marginRight: '10px' }} />
                        Teeth<span style={{ color: '#00bfa9' }}>Seg</span>
                    </h1>
                    <p style={styles.description}>
                        TeethSeg revolutionizes dental care with cutting-edge AI technology. Our advanced platform simplifies 3D tooth segmentation, providing faster and more precise results for orthodontic professionals. Step into the future of dental innovation with TeethSeg, where efficiency meets accuracy.
                    </p>
                    <br />
                    <button
                        style={{ ...styles.button, ...(isHovered ? styles.buttonHover : {}) }}
                        onMouseEnter={() => setIsHovered(true)}
                        onMouseLeave={() => setIsHovered(false)}
                        onClick={scrollToPanel} // Ajout de l'événement pour faire défiler vers le panel bleu
                    >
                        Get started &nbsp;&nbsp;>
                    </button>
                </div>
                <div style={styles.imageContainer}>
                    <img src={img_home} alt="3D Teeth Segmentation" style={styles.image} />
                </div>
            </section>

            {/* Section 2: Animated Panel */}
            <div id="blue-panel" style={styles.panel}> {/* Ajout de l'id "blue-panel" */}
                <div style={styles.textContainer}>
                    <p style={styles.animatedText}>
                        Ready for the takeoff? Dentbird puts wings on your solution.
                    </p>
                </div>
                <div style={styles.demoButtonContainer}>
                    <Link to="/upload" style={styles.demoButton}
                          onMouseEnter={(e) => (e.target.style.backgroundColor = '#e0e0e0')}
                          onMouseLeave={(e) => (e.target.style.backgroundColor = 'white')}
                    >
                        Book a demo
                    </Link>
                </div>

            </div>
        </div>
    );
};

export default Home;
