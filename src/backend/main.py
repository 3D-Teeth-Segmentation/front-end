
import os
import zipfile
from io import BytesIO

import vedo
import vtk
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

from meshsegnet import *
import numpy as np
import torch


def convert_vtp_to_obj_with_colors(vtp_file, obj_file, mtl_file, predicted_labels, num_classes):
    # Lire le fichier VTP
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()

    polydata = reader.GetOutput()

    # Écrire le fichier OBJ
    writer = vtk.vtkOBJWriter()
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetFileName(obj_file)

    # Définir les couleurs pour chaque classe
    colors = [
        (0.5, 0.5, 0.5),   # Gray
        (1.0, 0.0, 0.0),   # Red
        (0.0, 0.0, 1.0),   # Blue
        (0.0, 1.0, 0.0),   # Green
        (0.5, 0.0, 0.5),   # Purple
        (1.0, 0.65, 0.0),  # Orange
        (1.0, 0.75, 0.8),  # Pink
        (0.5, 0.5, 0.5),   # Gray (repeated)
        (1.0, 1.0, 0.0),   # Yellow
        (0.0, 0.5, 0.5),   # Teal
        (0.68, 0.85, 0.9), # Light Blue
        (0.0, 1.0, 0.0),   # Bright Green
        (1.0, 0.84, 0.0),  # Gold
        (0.65, 0.16, 0.16),# Brown
        (0.5, 0.0, 0.5),   # Dark Purple
        (0.4, 0.4, 0.4)    # Dark Gray
    ]

    with open(mtl_file, 'w') as mtl:
        for i in range(num_classes):
            r, g, b = colors[i]
            mtl.write(f"newmtl material_{i}\n")
            mtl.write(f"Ka {r} {g} {b}\n")  # Couleur ambiante
            mtl.write(f"Kd {r} {g} {b}\n")  # Couleur diffuse
            mtl.write(f"Ks {r} {g} {b}\n")  # Couleur spéculaire
            mtl.write("illum 2\n")

    # Référencer le fichier MTL dans l'OBJ
    with open(obj_file, 'w') as obj:
        obj.write(f"mtllib {mtl_file}\n")  # Référencer le fichier MTL

        for i in range(polydata.GetNumberOfPoints()):
            x, y, z = polydata.GetPoint(i)
            obj.write(f"v {x} {y} {z}\n")

        for i in range(polydata.GetNumberOfPolys()):
            ids = polydata.GetCell(i).GetPointIds()
            obj.write(f"usemtl material_{predicted_labels[i]}\n")
            obj.write(f"f {ids.GetId(0)+1} {ids.GetId(1)+1} {ids.GetId(2)+1}\n")


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the model
def load_model():
    model_path = 'models'
    model_name = 'Mesh_Segmentation_MeshSegNet_15_classes_72samples_best.tar'

    num_classes = 15
    num_channels = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device, dtype=torch.float)
    model.eval()

    return model, device


model, device = load_model()

print('model')


@app.post("/segmentation")
async def apply_segmentation(file: UploadFile = File(...)):
    print("Segmentation endpoint reached")
    if not os.path.exists('temp'):
        os.makedirs('temp')
    print("Uploading segmentation")
    if not file.filename.endswith('.obj'):
        return {"error": "File must be an OBJ format."}

    # Save the uploaded OBJ file
    print("Uploading")
    obj_path = f"temp/{file.filename}"
    with open(obj_path, "wb") as f:
        f.write(await file.read())

    # Convert OBJ to VTP
    vtp_path = obj_path.replace('.obj', '.vtp')
    mesh = vedo.load(obj_path)
    vedo.write(mesh, vtp_path)

    # Run the segmentation process
    print('Predicting Sample filename: {}'.format(vtp_path))

    # Process the mesh for prediction
    points = mesh.vertices
    mean_cell_centers = mesh.center_of_mass()
    points[:, 0:3] -= mean_cell_centers[0:3]

    ids = np.array(mesh.cells)
    cells = points[ids].reshape(mesh.ncells, 9).astype(dtype='float32')
    mesh.compute_normals()
    normals = mesh.celldata['Normals']
    barycenters = mesh.cell_centers
    barycenters -= mean_cell_centers[0:3]

    # Normalize data
    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    for i in range(3):
        cells[:, i] = (cells[:, i] - means[i]) / stds[i]
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

    X = np.column_stack((cells, barycenters, normals))

    # Computing A_S and A_L
    A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
    A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
    D = distance_matrix(X[:, 9:12], X[:, 9:12])
    A_S[D < 0.1] = 1.0
    A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

    A_L[D < 0.2] = 1.0
    A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

    # Numpy -> torch.tensor
    X = X.transpose(1, 0)
    X = X.reshape([1, X.shape[0], X.shape[1]])
    X = torch.from_numpy(X).to(device, dtype=torch.float)
    A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
    A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
    A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
    A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

    tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
    patch_prob_output = tensor_prob_output.detach().cpu().numpy()
    predicted_labels = np.argmax(patch_prob_output[0, :], axis=-1)

    # Output predicted labels
    mesh2 = mesh.clone()
    mesh2.celldata['MaterialIds'] = predicted_labels

    # Save segmented mesh as VTP
    segmented_vtp_path = os.path.join('temp', f'Sample_{os.path.basename("")}_predicted.vtp')
    vedo.write(mesh2, segmented_vtp_path, binary=True)
    print(segmented_vtp_path)

    # Convert VTP to OBJ with MTL for colors
    segmented_obj_path = os.path.join('temp', f'Sample_{os.path.basename("")}_predicted.obj')
    mtl_path = segmented_obj_path.replace('.obj', '.mtl')
    convert_vtp_to_obj_with_colors(segmented_vtp_path, segmented_obj_path, mtl_path, predicted_labels, 15)

    # Create a zip file containing the obj and mtl files
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.write(segmented_obj_path, os.path.basename(segmented_obj_path))
        zip_file.write(mtl_path, os.path.basename(mtl_path))

    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=segmented_files.zip"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
