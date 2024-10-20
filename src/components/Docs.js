import { AppWindow, Database, Download } from "lucide-react";
import { useState } from "react";

function Docs() {
    const [frontend, setFrontend] = useState(true);
    const [backend, setBackend] = useState(false);
    const [loading, setLoading] = useState(false);
    const [pdfUrl, setPdfUrl] = useState(""); // État pour stocker l'URL du PDF
    const cn = (...classNames) => classNames.filter(Boolean).join(" ");

    // Définir les chemins des fichiers
    const frontEndDocs = {
        filename: "FrontEnd Documentation BARBARA_ZERHERI",
        path: "/documents/FrontEnd Documentation Racha_Barbara Zerheri_FatimaZahra ENSET_SDIA Sujet_1_3dsf.pdf" // Mettez le bon chemin ici
    };

    const backEndDocs = {
        filename: "BackEnd Documentation BARBARA_ZERHERI",
        path: "/documents/Backend Documentation Racha_Barbara Zerheri_FatimaZahra ENSET_SDIA Sujet_1_3dsf .pdf" // Mettez le bon chemin ici
    };

    const handleDownload = (doc) => {
        setLoading(true);

        setTimeout(() => {
            const link = document.createElement('a');
            link.href = doc.path; // Utiliser le chemin du document
            link.target = '_blank';
            link.download = `${doc.filename}.pdf`;
            link.click();
            setLoading(false);
        }, 2000);
    };

    const handleShowPdf = (doc) => {
        setPdfUrl(doc.path); // Mettre à jour l'URL du PDF pour l'afficher
    };

    return (
        <div className="flex-box flex-col" >
            <div className="text-white h-full w-full py-10 scroll-smooth bg-primary-background">
                <div>
                    <div className="text-center flex-box flex-col md:flex-row gap-12 w-full">
                        <button
                            onClick={() => {
                                setFrontend(true);
                                setBackend(false);
                                handleShowPdf(frontEndDocs); // Afficher le PDF front-end
                            }}
                            className={cn("bg-slate-900 font-semibold py-4 px-8 hover:bg-slate-100 hover:text-slate-900 leading-tight rounded-lg transition ease-linear", frontend ? "bg-white text-slate-900 hover:bg-white" : "")}
                        >
                            <div className="flex items-center whitespace-nowrap">
                                <AppWindow className="mx-2" />
                                Front End Docs
                            </div>
                        </button>
                        <button
                            onClick={() => {
                                setFrontend(false);
                                setBackend(true);
                                handleShowPdf(backEndDocs); // Afficher le PDF front-end
                            }}
                            className={cn("bg-slate-900 font-semibold py-4 px-8 hover:bg-background hover:text-black leading-tight rounded-lg transition ease-linear", backend ? "bg-white text-black hover:bg-white" : "text-white")}
                        >
                            <div className="flex items-center whitespace-nowrap">
                                <Database className="mx-2" />
                                Back End Docs
                            </div>
                        </button>
                    </div>
                </div>

                <div className="lg:px-72 px-4">
                    {frontend &&
                        <>
                            <div className="flex flex-col md:flex-row justify-between">
                                <div className="flex mt-12 ml-2">
                                    <AppWindow className="mr-4 text-blue-500" size={42} />
                                    <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-500 to-green-400 bg-clip-text text-transparent">Front End Documentation</h2>
                                </div>

                                <div className="flex justify-between flex-col mt-12">
                                    <button
                                        onClick={() => { handleDownload(frontEndDocs); }} // Passer l'objet frontEndDocs
                                        className={cn("text-white bg-gradient-to-br from-green-400 to-blue-600 hover:bg-gradient-to-bl font-medium rounded-lg text-sm px-5 py-2.5 text-center mr-2 mb-2", loading ? "hover:bg-gradient-to-bl " : "bg-slate-900")}
                                    >
                                        <div className="flex items-center">
                                            <Download className="mx-2" />
                                            {loading ? "Downloading the file..." : "Download as PDF"}
                                        </div>
                                    </button>
                                </div>
                            </div>

                            {/* Afficher le PDF ici */}
                            {pdfUrl && (
                                <iframe
                                    src={pdfUrl}
                                    width="100%"
                                    height="600"
                                    className="mt-10 border border-gray-300"
                                    title="PDF Viewer"
                                ></iframe>
                            )}
                        </>
                    }

                    {backend &&
                        <>
                            <div className="flex flex-col md:flex-row justify-between">
                                <div className="flex mt-12 ml-2">
                                    <Database className="mr-4 text-blue-500" size={42} />
                                    <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-500 to-green-400 bg-clip-text text-transparent">Back End Documentation</h2>
                                </div>

                                <div className="flex justify-between flex-col mt-12">
                                    <button
                                        onClick={() => { handleDownload(backEndDocs); }} // Passer l'objet backEndDocs
                                        className={cn("text-white bg-gradient-to-br from-green-400 to-blue-600 hover:bg-gradient-to-bl font-medium rounded-lg text-sm px-5 py-2.5 text-center mr-2 mb-2", loading ? "hover:bg-gradient-to-bl" : "bg-slate-900")}
                                    >
                                        <div className="flex items-center whitespace-nowrap">
                                            <Download className="mx-2" />
                                            {loading ? "Downloading the file..." : "Download as PDF"}
                                        </div>
                                    </button>
                                </div>
                            </div>

                            {/* Afficher le PDF ici pour le back-end */}
                            {pdfUrl && (
                                <iframe
                                    src={pdfUrl}
                                    width="100%"
                                    height="600"
                                    className="mt-10 border border-gray-300"
                                    title="PDF Viewer"
                                ></iframe>
                            )}
                        </>
                    }
                </div>
            </div>
        </div>
    );
}

export default Docs;
