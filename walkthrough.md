# Geometric Neural Operator Walkthrough

## Overview
This project implements a Geometric Neural Operator (GNO) framework for learning PDEs on Riemannian (Sphere) and Lorentzian (Minkowski) manifolds. It includes a Python-based core library, trained models, and a React-based visualization UI.

## Project Structure
- `src/geometry/`: Manifold definitions (Sphere, Minkowski).
- `src/spectral/`: Spectral transforms (Spherical Harmonics, Fourier).
- `src/models/`: Neural Network components (Branch, Trunk, GNO).
- `src/physics/`: PDE definitions and Loss functions.
- `ui/backend/`: FastAPI server serving trained models.
- `ui/frontend/`: React + Three.js visualization for inference results.

## Verification
To verify the installation and core logic, run:
```bash
python verify_installation.py
```

## Training Models
To train the models for Sphere (Poisson) and Minkowski (Wave) equations:
```bash
python train.py
```
This will save `sphere_model.pth` and `minkowski_model.pth` in the project root.

## Running the Application

### 1. Start the Backend
The backend loads the trained models and serves predictions.
```bash
cd ui/backend
uvicorn app:app --reload
```
Server runs at `http://localhost:8000`.

### 2. Start the Frontend
The frontend visualizes the manifolds and fields.
```bash
cd ui/frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

## Features
- **Sphere Visualization**: 3D interactive sphere with heatmap of the predicted field $u$.
- **Minkowski Visualization**: 2D spacetime grid (Time vs Space) visualized in 3D, showing the wave propagation.
- **Real-time Inference**: Click "Run New Inference" to generate new random inputs and see the model's prediction.

## Next Steps
- **Hyperbolic Geometry**: Implement the Hyperboloid model in `src/geometry/hyperbolic.py`.
- **Advanced Physics**: Add more complex PDEs (e.g., Navier-Stokes on Sphere).
- **Causality**: Refine the causality loss in `src/physics/loss.py`.
