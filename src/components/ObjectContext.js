import React, { createContext, useState, useContext } from 'react';

// Créer le contexte
const ObjectContext = createContext();

// Créer un provider pour le contexte
export const ObjectProvider = ({ children }) => {
    const [object, setObject] = useState(null);

    return (
        <ObjectContext.Provider value={{ object, setObject }}>
            {children}
        </ObjectContext.Provider>
    );
};

// Hook pour utiliser le contexte
export const useObject = () => useContext(ObjectContext);
