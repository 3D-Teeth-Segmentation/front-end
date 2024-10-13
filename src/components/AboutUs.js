import React from 'react';
import { Container, Typography, Box, Avatar, Grid, Paper } from '@mui/material';
import { LinkedIn, GitHub } from '@mui/icons-material'; // Icons for LinkedIn and GitHub
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import GitHubIcon from '@mui/icons-material/GitHub';

// Import your technology logos like this
import reactLogo from '../images/react-logo.png';
import muiLogo from '../images/mui-logo.png';
import pythonLogo from '../images/python-50.png';
import logo1 from '../images/fatima_image.jpg'; // Use your uploaded image path
import logo2 from '../images/racha_image.jpeg';
import jsLogo from '../images/icons8-js-50.png';

function AboutUs() {
    const teamMembers = [
        {
            name: 'Fatima zahrae Zerheri',
            description: "Data science & IA student",
            school: "(ENSET)",
            image: logo1,
            linkedin: 'https://www.linkedin.com/in/fatima-zahrae-zerheri-26aa59298/',
            github: 'https://github.com/ZerheriFatimaZahrae',
        },
        {
            name: 'Racha Barbara',
            description: "Data science & IA student",
            school: "(ENSET)",
            image: logo2,
            linkedin: 'https://www.linkedin.com/in/racha-barbara-b8b92a223/',
            github: 'https://github.com/BarbaraRacha',
        },
    ];

    return (
        <Container sx={{ marginTop: 8 }}>
            <Typography
                variant="h4"
                gutterBottom
                textAlign="center"
                sx={{
                    fontFamily: 'Verdana',
                    fontWeight: 'bold',
                    fontSize: 30,
                    color: '#333'
                }}
            >
                About Us
            </Typography>

            <Grid container spacing={4} justifyContent="center" sx={{ marginTop: 4 }}>
                {teamMembers.map((member, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                        <Paper elevation={3} sx={{ padding: 4, textAlign: 'center' }}>
                            <Avatar
                                alt={member.name}
                                src={member.image}
                                sx={{ width: 80, height: 80, margin: '0 auto' }}
                            />
                            <Typography variant="h6" sx={{ marginTop: 2, fontFamily: 'Verdana' }}>
                                {member.name}
                            </Typography>
                            <Typography variant="body1" sx={{ marginTop: 1, fontFamily: 'Verdana', fontSize: 14 }}>
                                {member.description}
                            </Typography>
                            <Typography variant="body2" sx={{ fontFamily: 'Verdana', fontSize: 14 }}>
                                {member.school}
                            </Typography>
                            <Box sx={{ marginTop: 2 }}>
                                <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="icon">
                                    <LinkedInIcon sx={{ marginRight: 1 }} />
                                </a>
                                <a href={member.github} target="_blank" rel="noopener noreferrer" className="icon">
                                    <GitHubIcon />
                                </a>
                            </Box>
                        </Paper>
                    </Grid>
                ))}
            </Grid>

            {/* Technologies Used Section */}
            <Typography variant="h5" gutterBottom textAlign="center" sx={{ marginTop: 15 }}>
                Technologies Used
            </Typography>
            <Grid container spacing={2} justifyContent="center" sx={{ marginTop: 2 }}>
                <Grid item>
                    <Avatar alt="React" src={reactLogo} sx={{ width: 56, height: 56 }} />
                    <Typography align="center">React</Typography>
                </Grid>
                <Grid item>
                    <Avatar alt="Material-UI" src={muiLogo} sx={{ width: 56, height: 56 }} />
                    <Typography align="center">Material-UI</Typography>
                </Grid>
                <Grid item>
                    <Avatar alt="Three.js" src={pythonLogo} sx={{ width: 56, height: 56 }} />
                    <Typography align="center">Python</Typography>
                </Grid>
                <Grid item>
                    <Avatar alt="Three.js" src={jsLogo} sx={{ width: 56, height: 56 }} />
                    <Typography align="center">JavaScript</Typography>
                </Grid>
            </Grid>
        </Container>
    );
}

export default AboutUs;
