import React from 'react';
import { Box, Typography, IconButton } from '@mui/material';
import FacebookIcon from '@mui/icons-material/Facebook';
import InstagramIcon from '@mui/icons-material/Instagram';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import TwitterIcon from '@mui/icons-material/Twitter';
import logo from '../images/logo.png';


function Footer() {
    return (
        <Box
            sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '10px 40px',
                backgroundColor: '#f8f9fa',
                borderTop: '1px solid #ddd',
                height: '60px',
                position: 'fixed', // Fixe le Footer
                bottom: 0, // Positionne le Footer en bas
                left: 0, // Aligne le Footer à gauche
                right: 0, // Aligne le Footer à droite
            }}
        >
            {/* Logo and text */}
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <img
                    src={logo}
                    alt="TeethSeg Logo"
                    style={{ width: '30px', marginRight: '10px' }}
                />
                <Typography variant="h6" component="span" sx={{ fontWeight: 'bold' }}>
                    Teeth<span style={{ color: '#00bfa5' }}>Seg</span>
                </Typography>
            </Box>

            {/* Copyright */}
            <Typography variant="body2" color="textSecondary">
                © 2024 TeethSeg - All Rights Reserved by 3DSF Interns
            </Typography>

            {/* Social media icons */}
            <Box>
                <IconButton href="https://www.facebook.com" target="_blank">
                    <FacebookIcon />
                </IconButton>
                <IconButton href="https://www.instagram.com" target="_blank">
                    <InstagramIcon />
                </IconButton>
                <IconButton href="https://www.linkedin.com" target="_blank">
                    <LinkedInIcon />
                </IconButton>
                <IconButton href="https://www.twitter.com" target="_blank">
                    <TwitterIcon />
                </IconButton>
            </Box>
        </Box>
    );
}

export default Footer;
