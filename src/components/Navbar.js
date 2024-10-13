import React, { useState, useEffect } from 'react';
import { AppBar, Toolbar, Typography, Box, Button, IconButton, Menu, MenuItem } from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';
import logo from '../images/logo.png';
import { Home, Description, Info, ContactMail } from '@mui/icons-material';
import { Link } from 'react-router-dom';

function Navbar({ themeMode, changeThemeMode }) { // Recevoir changeThemeMode comme prop
    const [anchorEl, setAnchorEl] = useState(null);
    const open = Boolean(anchorEl);

    const handleMenuClick = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
    };

    useEffect(() => {
        const savedTheme = localStorage.getItem('theme') || 'light';
        changeThemeMode(savedTheme); // Appliquer le thème au chargement
        document.body.classList.toggle('dark', savedTheme === 'dark'); // Ajouter ou retirer la classe 'dark'
    }, [changeThemeMode]);

    const handleThemeChange = (mode) => {
        changeThemeMode(mode);
        handleMenuClose();
    };

    return (
        <AppBar
            position="static"
            sx={{ backgroundColor: themeMode === 'dark' ? '#20232a' : '#f6f842' }} // Couleur navbar
        >
            <Toolbar sx={{ height: 70 }}>
                {/* Logo */}
                <Box
                    component="img"
                    sx={{ height: 40, marginRight: 2 }}
                    alt="Logo"
                    src={logo}
                />

                {/* Texte "TeethSeg" */}
                <Typography
                    variant="h6"
                    sx={{
                        color: themeMode === 'dark' ? '#fff' : '#333', // Couleur selon le mode
                        fontWeight: 'bold',
                        fontSize: 25,
                        fontFamily: 'verdana',
                        flexGrow: 1, // Prendre tout l'espace pour centrer
                    }}
                >
                    Teeth<span style={{ color: '#00bfa9' }}>Seg</span>
                </Typography>

                <div>
                    {/* Boutons de navigation */}
                    <Button sx={{ color: themeMode === 'dark' ? '#fff' : '#333', fontSize: 18, marginLeft: -100, fontFamily: 'Verdana' }}>
                        <Home sx={{ marginRight: 1 }} />
                        <Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>
                            Home
                        </Link>
                    </Button>
                    <Button sx={{ color: themeMode === 'dark' ? '#fff' : '#333', fontSize: 18, marginLeft: 10, fontFamily: 'Verdana' }}>
                        <Description sx={{ marginRight: 1 }} /> Docs
                    </Button>
                    <Button sx={{ color: themeMode === 'dark' ? '#fff' : '#333', fontSize: 18, marginLeft: 10, fontFamily: 'Verdana' }}>
                        <Info sx={{ marginRight: 1 }} />
                        <Link to="/about" style={{ color: 'inherit', textDecoration: 'none' }}>
                            About
                        </Link>
                    </Button>

                    <Button sx={{ color: themeMode === 'dark' ? '#fff' : '#333', fontSize: 18, marginLeft: 10, fontFamily: 'Verdana' }}>
                        <ContactMail sx={{ marginRight: 1 }} />
                        <Link to="/contact" style={{ color: 'inherit', textDecoration: 'none' }}> {/* Ensure proper link styling */}
                            Contact
                        </Link>
                    </Button>
                </div>

                {/* Bouton de connexion */}
                <Button
                    sx={{
                        color: themeMode === 'dark' ? '#fff' : '#333',
                        backgroundColor: themeMode === 'dark' ? '#333' : '#fff',
                        borderRadius: '20px',
                        padding: '5px 20px',
                        marginRight: '10px',
                    }}
                >
                    Sign In
                </Button>

                {/* Bouton pour basculer entre mode clair/sombre avec menu */}
                <IconButton
                    sx={{ ml: 1, color: themeMode === 'dark' ? '#fff' : '#333' }}
                    onClick={handleMenuClick}
                >
                    {themeMode === 'dark' ? <Brightness7 /> : <Brightness4 />}
                </IconButton>
                <Menu
                    anchorEl={anchorEl}
                    open={open}
                    onClose={handleMenuClose}
                >
                    <MenuItem onClick={() => handleThemeChange('light')}>Light</MenuItem>
                    <MenuItem onClick={() => handleThemeChange('dark')}>Dark</MenuItem>
                    <MenuItem onClick={() => handleThemeChange('system')}>System</MenuItem>
                </Menu>
            </Toolbar>
        </AppBar>
    );
}

export default Navbar;