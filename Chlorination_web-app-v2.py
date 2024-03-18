import streamlit as st
from streamlit_ketcher import st_ketcher
st.set_page_config(layout="wide")

# Data loading and management
import pandas as pd
import pickle
import numpy as np


# Basic chemistry packages
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions as Reactions

# A few settings to improve the quality of structures
from rdkit.Chem import rdDepictor

IPythonConsole.ipython_useSVG = True
rdDepictor.SetPreferCoordGen(True)



# Non-standard metrics
from sklearn.metrics import matthews_corrcoef as MCC

# Plotting packages
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 96})
plt_parameters = plt.rcParams
plt.rcParams.update(plt_parameters)  # to reset parameters back to plt from sns
import seaborn as sns

sns.set_theme(style="ticks")

# Fingerprints functions
fp_size = [2048]
fp_radius = [2]


def smiles_to_fp(smiles_list):
    fp_list = []
    for smile in smiles_list:
        fp_list.append(fp_as_array(smile))
    return fp_list


def fp_as_array(smile):
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius[0], nBits=fp_size[0])
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


st.title("Prediction of chlorination site")

models_FP2FP = {}

models_SVD2SVD = {}


def possible_prods(starting_material):
    """
    Input: RDKit Mol-object of starting material

    Output: List of SMILES of each mono-chlorination combination in aromatic system.
            If no aromatic system is recognized, it returns from all CH bonds that might become C-Cl
    """

    potential_products_Ar = []
    molecule = Chem.AddHs(starting_material)

    for atom in molecule.GetAtoms():
        pot_product = molecule

        if atom.GetSymbol() == "H":
            for neighbour in atom.GetNeighbors():
                if neighbour.GetIsAromatic() and neighbour.GetAtomicNum() == 6:
                    atom.SetAtomicNum(17)
                    potential_products_Ar.append(Chem.MolToInchi(pot_product))
                    atom.SetAtomicNum(1)

    if len(potential_products_Ar) == 0:  # In case RDKit does not manage to find an aromatic ring
        for atom in molecule.GetAtoms():
            pot_product = molecule

            if atom.GetSymbol() == "H":
                for neighbour in atom.GetNeighbors():
                    if neighbour.GetAtomicNum() == 6:
                        atom.SetAtomicNum(17)
                        potential_products_Ar.append(Chem.MolToInchi(pot_product))
                        atom.SetAtomicNum(1)

    potential_products_Ar = set(potential_products_Ar)
    potential_products = [Chem.MolFromInchi(molecule) for molecule in potential_products_Ar]
    potential_products = [Chem.MolToSmiles(molecule) for molecule in potential_products]
    return potential_products

def count_halogenations(mol):
    molecule = Chem.AddHs(mol)
    nb_halogens = 0
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == 'H':
            for neighbour in atom.GetNeighbors():
                if neighbour.GetIsAromatic() and neighbour.GetAtomicNum() == 6:
                    nb_halogens += 1
    return nb_halogens
def compare_FP_to_products_MCC(pred_FP, products):
    """
    Input
     - pred_FP: numpy-array of the FP predicted to be the product
     - products: List of SMILES of possible products

    Selection of the most likely product in this function is based on the best FP-MCC

     Output
      - Product FP (np.array)
      - MCC between the predicted and selected FP
      - SMILES of the selected product
    """
    prods_FP = smiles_to_fp(products)
    # prods_bit = [DataStructs.cDataStructs.CreateFromBitString("".join(prod_FP.astype(str))) for prod_FP in prods_FP]
    similarity_results = []
    for prod_FP in prods_FP:
        ## Change the similarity metric to MCC between fingerprints
        similarity_results.append(MCC(prod_FP, pred_FP))

    most_similar = similarity_results.index(max(similarity_results))

    return prods_FP[most_similar], similarity_results[most_similar], products[most_similar]

def load_models_old():
    return pickle.load(open("model_fp2fp_Ridge_7.5.mlpickle", "rb"))
@st.cache_resource
def load_models():
    return pickle.load(open("superset_model.pkl", "rb"))

try:
    fp2fp_model = load_models()
except:
    st.error("No model available")


def run_model_fp(model_key, mol):
    if model_key == "Your Choice":
        return selected_product
    else:
        model = fp2fp_model

        reactant = Chem.MolToSmiles(mol)
        reactant_FP = smiles_to_fp([reactant])
        predictions_FP = model.predict(reactant_FP)[0]
        if len(np.unique(predictions_FP)) != 2:
            predictions_FP = np.digitize(predictions_FP, bins=[0.5, 3])


        predicted_product = compare_FP_to_products_MCC(predictions_FP, possible_prods(
            Chem.MolFromSmiles(reactant)))[2]


        return [predicted_product]#, similarity


intro_col1, intro_col2 = st.columns(2)

with intro_col1:
    st.markdown("This app predicts the most likely product of a halogenation reaction on an aromatic systems. "
                "The AI model generates all possible halogenation products, and the predicted Morgan Fingerprint, "
                "then it matches the product to the most similar FP (Selected product)."
                "Below you can see a general scheme of the workflow: ")
with intro_col2:
    st.markdown("This model was generated using a data-only approach, i.e. no prior knowledge of the reaction mechanism was used. "
                "To see it working, input the SMILES of the molecule you are interested in, then run the model.")

st.image("workflow.png", use_column_width=True)

st.markdown("Disclaimer: No data is stored in this version of the app.")
st.markdown("----------------------------------------------------------------")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.markdown('Enter the molecule:')
    input_type = st.radio("Select your input method:", ("Draw it", "SMILES", "SDF file"))

    if input_type == "Draw it":
        smiles_string = st_ketcher("Oc1ccccc1")

    elif input_type == "SMILES":

        smiles_string = st.text_input('SMILES String')

    elif input_type == "SDF file":
        sdf_file = st.file_uploader("Upload SDF file", type=["sdf"])
        if sdf_file is not None:
            sdf_smiles = LoadSDF(sdf_file, smilesName='SMILES')["SMILES"]
            entry = 0
            if len(sdf_smiles) > 1:
                st.markdown("More than one molecule in file. Select the one to be used:")
                entry = st.slider("Select molecule", 1, len(sdf_smiles), step = 1) - 1
            smiles_string = sdf_smiles[entry]

    mol = Chem.MolFromSmiles(smiles_string)

    try:
        mol = Chem.MolFromSmiles(smiles_string)

        if mol is None:
            raise ValueError("Invalid Molecule")
        elif len(smiles_string) == 0:
            raise ValueError("No molecule found")
        elif count_halogenations(mol) == 0:
            raise ValueError("No aromatic substitution site found in molecule")

    except ValueError as e:
        st.error(str(e))
        st.stop()


with col2:

    st.markdown("#### Predicted chlorination product")

    if count_halogenations(mol) > 1:
        nb_halogens = st.select_slider("Number of halogenations", options=list(range(1, count_halogenations(mol)+1)), value=1)
    else:
        nb_halogens = 1
    st.write("Number of halogenations: ", nb_halogens)
    products = []
    products_image = []


    potential_products = [Chem.MolToSmiles(mol)]
    products_list = [mol]

    progress_bar = st.progress(0)
    for i in range(nb_halogens):
        products = []
        for potential_product in potential_products:
            for new_possible_prods in possible_prods(Chem.MolFromSmiles(potential_product)):
                #st.write(new_possible_prods)
                products.append(new_possible_prods)
        potential_products = products
        #Here!
        products_list.append(Chem.MolFromSmiles(run_model_fp("Ridge", products_list[i])[0]))
        progress_bar.progress((i+1)/nb_halogens, "Progress: " + str(i+1) + "/" + str(nb_halogens) + " halogenations")
    predicted_product = products_list[-1]



    products_inchi = []
    for product in products:
        products_inchi.append(Chem.MolToInchi(Chem.MolFromSmiles(product)))

    products_mol = []
    for product in list(set(products_inchi)):
        products_mol.append(Chem.MolFromInchi(product))
    products = products_mol

    for potential_product in products:
        products_image.append(Draw.MolToImage(potential_product, size=(100, 100)))


    predictions = {"ML prediction": predicted_product}

    st.image(Draw.MolToImage(predictions["ML prediction"]))
    st.markdown("Product SMILES: " + Chem.MolToSmiles(predicted_product))


    model_selection = []

    for idx in range(len(products)):
        model_pick = []
        for model in predictions.keys():
            if Chem.MolToInchi(products[idx]) == Chem.MolToInchi(predictions[model]):
                model_pick.append(f'{model} \n')
                # Append to the final list which will go into the dict
                model_selection.append(model_pick)
            else:
                model_selection.append("")


    if len(potential_products) == 0:
        raise ValueError('No possible products found')



rxn_string = Chem.MolToSmiles(mol)
for molecule in products_list[1:]:
    rxn_string += "." + Chem.MolToSmiles(molecule)

st.markdown("#### Reaction scheme")
rxn_schema = st.columns([0.5, 2*len(products_list), 0.5])[1]
with rxn_schema:
    st.image(Draw.MolsToImage(products_list, molsPerRow=len(products_list), subImgSize=(200, 200)), use_column_width=False)


# Model outputs!
st.markdown("----------------------------------------------------------------")

st.markdown("### Possible products list")

col1, col2, col3 = st.columns(3)

data = {'Text': [f'Product  {i+1}' for i in range(len(products))],
        'Product': products_image,
        'Models': model_selection
        }

df = pd.DataFrame(data)

for index, row in df.iterrows():

    if index % 3 == 0:
        with col1:
            st.write(row['Text'])
            st.image(row['Product'], use_column_width=False)
            st.write(row['Models'])

    elif index % 3 == 1:
        with col2:
            st.write(row['Text'])
            st.image(row['Product'], use_column_width=False)
            st.write(row['Models'])

    elif index % 3 == 2:
        with col3:
            st.write(row['Text'])
            st.image(row['Product'], use_column_width=False)
            st.write(row['Models'])

