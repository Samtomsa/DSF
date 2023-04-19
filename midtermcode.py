from pathlib import Path
from warnings import filterwarnings
!pip install rdkit
# Silence some expected warnings
filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# Neural network specific libraries
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

HERE = Path(_dh[-1])
DATA1 = HERE / "kinase.csv"
DATA2 = HERE / "best_weights.hdf5"

def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.
    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.
    method : str
        The type of fingerprint to use. Default is MACCS keys.
    n_bits : int
        The length of the fingerprint.
    Returns
    -------
    array
        The fingerprint array.
    """

    # Convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        return np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    if method == "morgan3":
        return np.array(GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits))
    else:
        print(f"Warning: Wrong method specified: {method}." " Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))
  
  def neural_network_model(hidden1, hidden2):
    """
    Creating a neural network from two hidden layers
    using ReLU as activation function in the two hidden layers
    and a linear activation in the output layer.
    Parameters
    ----------
    hidden1 : int
        Number of neurons in first hidden layer.
    hidden2: int
        Number of neurons in second hidden layer.
    Returns
    -------
    model
        Fully connected neural network model with two hidden layers.
    """

    model = Sequential()
    # First hidden layer
    model.add(Dense(hidden1, activation="relu", name="layer1"))
    # Second hidden layer
    model.add(Dense(hidden2, activation="relu", name="layer2"))
    # Output layer
    model.add(Dense(1, activation="linear", name="layer4"))
    
    # Compile model
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=["mse", "mae"])
    return model
    
# url = "https://cloud-new.gdb.tools/index.php/s/ZfZM7itQf3rm6Sw/download"
# df = pd.read_csv(url, index_col=0)
# df = df.reset_index(drop=True)

df = pd.read_csv(DATA1, index_col=0)
df = df.reset_index(drop=True)


# Keep necessary columns
chembl_df = df[["smiles", "target_chembl_id","standard_value"]]
# Convert IC50 to pIC50 (standart_value to the pIC50)
chembl_df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)

chembl_df.replace([np.inf, -np.inf], np.nan, inplace=True)
chembl_df.dropna(inplace=True)

chembl_df["fingerprints_df"] = chembl_df["smiles"].apply(smiles_to_fp)

# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(
    chembl_df["fingerprints_df"], chembl_df[["pIC50"]], test_size=0.3, random_state=42
)
# Neural network parameters
batch_sizes = [16, 32, 64]
nb_epoch = 50
layer1_size = 64
layer2_size = 32

# Save the trained model

checkpoint = ModelCheckpoint(
    str(DATA2),
    monitor="loss",
    verbose=0,
    save_best_only=True,
    mode="min",
    save_weights_only=True,
)
callbacks_list = [checkpoint]

# Fit the model
model.fit(
    np.array(list((x_train))).astype(float),
    y_train.values,
    epochs=nb_epoch,
    batch_size=16,
    callbacks=callbacks_list,
    verbose=0,
)
