from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.geometry.sphere import Sphere
from src.geometry.minkowski import Minkowski
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.gno import GeometricDeepONet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ManifoldRequest(BaseModel):
    type: str
    params: dict = {}

@app.get("/")
def read_root():
    return {"message": "Geometric Neural Operator Backend"}

@app.post("/manifold/info")
def get_manifold_info(req: ManifoldRequest):
    if req.type == "sphere":
        return {"type": "sphere", "dim": 2, "embedding_dim": 3, "description": "S^2 embedded in R^3"}
    elif req.type == "minkowski":
        return {"type": "minkowski", "dim": 2, "embedding_dim": 2, "description": "1+1D Minkowski Spacetime"}
    else:
        raise HTTPException(status_code=400, detail="Unknown manifold type")

@app.get("/benchmark/results")
def get_benchmark_results():
    # Load results from JSON
    results_path = os.path.join(os.path.dirname(__file__), "../../benchmark_results.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, "r") as f:
            data = json.load(f)
        return {"status": "success", "data": data}
    else:
        return {"status": "error", "message": "Benchmark results not found. Run benchmark.py first."}

@app.get("/benchmark/plot")
def get_benchmark_plot():
    # Serve the plot image
    from fastapi.responses import FileResponse
    plot_path = os.path.join(os.path.dirname(__file__), "../../convergence_plot.png")
    if os.path.exists(plot_path):
        return FileResponse(plot_path)
    else:
        raise HTTPException(status_code=404, detail="Plot not found")

@app.post("/inference")
def run_inference(req: ManifoldRequest):
    if req.type not in models:
        return {"status": "error", "message": "Model not loaded or training incomplete."}
    
    model = models[req.type]
    
    with torch.no_grad():
        if req.type == "sphere":
            f_features = torch.randn(1, 128).to(device)
            nlat, nlon = 16, 32
            theta = torch.linspace(0, np.pi, nlat)
            phi = torch.linspace(0, 2*np.pi, nlon)
            T, P = torch.meshgrid(theta, phi, indexing='ij')
            x = torch.sin(T) * torch.cos(P)
            y = torch.sin(T) * torch.sin(P)
            z = torch.cos(T)
            grid_flat = torch.stack([x, y, z], dim=-1).reshape(-1, 3).unsqueeze(0).to(device)
            u_pred = model(f_features, grid_flat)
            points = grid_flat.cpu().numpy().reshape(-1, 3)
            values = u_pred.cpu().numpy().reshape(-1)
            data = []
            for i in range(len(points)):
                data.append({"x": float(points[i, 0]), "y": float(points[i, 1]), "z": float(points[i, 2]), "val": float(values[i])})
            return {"status": "success", "data": data}
        elif req.type == "minkowski":
            u0 = torch.randn(1, 64).to(device)
            nt, nx = 20, 20
            t = torch.linspace(0, 1, nt)
            x = torch.linspace(0, 2*np.pi, nx)
            T, X = torch.meshgrid(t, x, indexing='ij')
            grid_flat = torch.stack([T, X], dim=-1).reshape(-1, 2).unsqueeze(0).to(device)
            u_pred = model(u0, grid_flat)
            points = grid_flat.cpu().numpy().reshape(-1, 2)
            values = u_pred.cpu().numpy().reshape(-1)
            data = []
            for i in range(len(points)):
                data.append({"t": float(points[i, 0]), "x": float(points[i, 1]), "val": float(values[i])})
            return {"status": "success", "data": data}
    return {"status": "error", "message": "Unknown error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
