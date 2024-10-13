// src/components/Contact.js
import React, { useState } from "react";
import SendIcon from '@mui/icons-material/Send';
import { Container, TextField, Button, Grid, Typography, Box } from '@mui/material';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import PhoneIcon from '@mui/icons-material/Phone';
import EmailIcon from '@mui/icons-material/Email';

const Contact = () => {
    const [formData, setFormData] = useState({
        email: "",
        subject: "",
        message: ""
    });

    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        try {
            // Here you can handle form submission, e.g., via an API
            console.log("Form data:", formData);
            // Example of a mock API call
            await new Promise(resolve => setTimeout(resolve, 1000));
            alert("Message sent successfully!");
        } catch (error) {
            console.error("Error submitting form:", error);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="contact-container" style={styles.container} >
            <h2 style={styles.header}>Contact Us</h2>
            <p style={styles.subtext}>
                Got a technical issue? Want to send feedback about a beta feature? Let us know.
            </p>
            <form onSubmit={handleSubmit} style={styles.form}>
                <input
                    type="email"
                    name="email"
                    placeholder="name@teethseg.com"
                    value={formData.email}
                    onChange={handleChange}
                    style={styles.input}
                    required
                />
                <input
                    type="text"
                    name="subject"
                    placeholder="Let us know how we can help you"
                    value={formData.subject}
                    onChange={handleChange}
                    style={styles.input}
                    required
                />
                <textarea
                    name="message"
                    placeholder="Leave a comment..."
                    value={formData.message}
                    onChange={handleChange}
                    style={styles.textarea}
                    required
                />
                <button type="submit" style={styles.button} disabled={isSubmitting} variant="contained"
                        color="primary"
                        fullWidth
                        endIcon={<SendIcon />}
                        >
                    {isSubmitting ? "Sending..." : "Send Message"}
                </button>
            </form>

            <Box textAlign="center" sx={{marginTop: 5}}>
                <Typography variant="h5">Feel free to get in touch with us !</Typography>
            </Box>

            <br/>

            <Box
                sx={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)', // Ensures two columns
                    justifyContent: 'space-around',
                    marginTop: '50px',
                    color: '#2d3748'
                }}
            >
                {/* Box for "Our Location" */}
                <Box sx={{ gridColumn: 1 }} textAlign="center">
                    <Box display="flex" justifyContent="center" alignItems="center">
                        <LocationOnIcon fontSize="medium" sx={{ color: '#2d3748', marginRight: '8px' }} />
                        <Typography variant="h6">&nbsp;Our Location&nbsp;</Typography>
                        <LocationOnIcon fontSize="medium" sx={{ color: '#2d3748', marginRight: '8px' }} />
                    </Box>
                    <Typography variant="subtitle2">
                        Villa Num 75 Lotissement la gare<br />
                        Mohammedia Maroc
                    </Typography>
                </Box>

                {/* Box for "How Can We Help?" */}
                <Box sx={{ gridColumn: 2 }} textAlign="center">
                    <Box display="flex" justifyContent="center" alignItems="center">
                        <PhoneIcon fontSize="medium" sx={{ color: '#2d3748', marginRight: '8px' }} />
                        <Typography variant="h6">&nbsp;How Can We Help &nbsp;</Typography>
                        <EmailIcon fontSize="medium" sx={{ color: '#2d3748', marginRight: '4px' }} />
                    </Box>
                    <Typography variant="subtitle2">
                        +212 5 23 30 04 46<br />
                        <Box display="flex" justifyContent="center" alignItems="center">
                            3dsmartfactory@gmail.com
                        </Box>
                    </Typography>
                </Box>
            </Box>
        </div>
    );
};

// Styles inline pour simplifier, tu peux les externaliser dans un fichier CSS si tu préfères
const styles = {
    container: {
        paddingTop: "5px",
        paddingBottom: "50px",
        paddingRight: "100px",
        paddingLeft: "100px",
        maxWidth: "1000px",
        margin: "0 auto",
        backgroundColor: "#f0f0f0",
        borderRadius: "8px",
        boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)" // Added a subtle shadow
    },
    header: {
        textAlign: "center",
        fontSize: 35,
        fontWeight: 'bold',
        background: 'linear-gradient(90deg, #212121, #68d391)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        marginBottom: '20px',
    },
    subtext: {
        textAlign: "center",
        marginBottom: "20px",
        fontSize: "1rem"
    },
    form: {
        display: "flex",
        flexDirection: "column",
        gap: "15px"
    },
    input: {
        padding: "10px",
        fontSize: "1rem",
        borderRadius: "4px",
        border: "1px solid #ccc",
        boxSizing: "border-box" // Ensures padding does not affect width
    },
    textarea: {
        padding: "10px",
        fontSize: "1rem",
        height: "100px",
        borderRadius: "4px",
        border: "1px solid #ccc",
        boxSizing: "border-box" // Ensures padding does not affect width
    },
    button: {
        backgroundColor: "#3182CE",
        color: "white",
        padding: "10px",
        borderRadius: "4px",
        border: "none",
        cursor: "pointer",
        fontSize: "1rem"
    },
    infoSection: {
        marginTop: "20px",
        textAlign: "center"
    },
    infoItem: {
        marginBottom: "10px"
    }
};

export default Contact;
