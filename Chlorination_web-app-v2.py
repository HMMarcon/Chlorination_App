import streamlit as st
from streamlit_ketcher import st_ketcher
st.set_page_config(layout="wide")

# Data loading and management
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

import numpy as np
from scipy.stats import norm

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

# Data pre-processing packages
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Regression Analysis
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

# Non-standard metrics
from sklearn.metrics import matthews_corrcoef as MCC

# Plotting packages
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 96})
plt_parameters = plt.rcParams
plt.rcParams.update(plt_parameters)  # to reset parameters back to plt from sns

# For interacting with molecules as widgets
# from ipywidgets import interact
# import ipywidgets as widgets

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate
import seaborn as sns

sns.set_theme(style="ticks")
from sklearn.metrics import make_scorer

from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge

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


# X_validation = pd.DataFrame("") #Validation set

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

# @st.cache_resource
# def load_svd():
#    return pickle.load(open("svd_transform.p", "rb"))

# svd2svd_models = load_models()
# svd = load_svd()
fp2fp_model = load_models()

def modified_z_score(data):
    """
    Calculates the modified Z-score for each observation in the data.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return modified_z

def z_score_to_probability(z_score):
    """
    Converts a Z-score to a probability of being an outlier from the standard normal distribution.
    """
    probability = 1 - norm.sf(abs(z_score)) * 2
    return probability

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
        #print(predictions_FP)

        predicted_product = compare_FP_to_products_MCC(predictions_FP, possible_prods(
            Chem.MolFromSmiles(reactant)))[2]
        predicted_product_FP = compare_FP_to_products_MCC(predictions_FP, possible_prods(
            Chem.MolFromSmiles(reactant)))[0]
        prods_FP = smiles_to_fp(possible_prods(Chem.MolFromSmiles(reactant)))
        # prods_bit = [DataStructs.cDataStructs.CreateFromBitString("".join(prod_FP.astype(str))) for prod_FP in prods_FP]
        similarity_mcc = []
        similarity = []

        #if len(prods_FP) == 1:
        #    similarity = [MCC(prods_FP, predictions_FP)]

        #elif len(prods_FP) == 2:
            #Normalised MCC for two products. No real statistical meaning, just shows how much distribution favours one.
        #    for prod_FP in prods_FP:
        #        similarity.append(MCC(prod_FP, predictions_FP))
            #similarity = similarity / sum(similarity)
        #elif len(prods_FP) > 2:

        #    for prod_FP in prods_FP:
                ## Change the similarity metric to MCC between fingerprints
        #        similarity_mcc.append(MCC(prod_FP, predictions_FP))
                #similarity = (similarity - sum(similarity)/len(similarity)) / (max(similarity) - min(similarity))
            #post-process similarity metrics to be between 0 and 1

            #Should just do a Student's t-test? Sure!
            #Former idea of a similarity metric based on a modified Z-score. Not used anymore.
        #    #similarity_mod_Z = modified_z_score(similarity_mcc)
        #    #similarity_proba = z_score_to_probability(similarity_mod_Z)
        #    #similarity = similarity_proba
        #    #similarity = (similarity - min(similarity)) / (max(similarity) - min(similarity))
        #    #similarity = similarity * similarity_mcc


        #for mcc_base in similarity_mcc:
        #    index = similarity_mcc.index(mcc_base)
        #    st.write(MCC(prods_FP[index],predicted_product_FP))
        #   st.write(mcc_base)
        #    similarity.append(mcc_base / MCC(prods_FP[index],predicted_product_FP))

        return [predicted_product]#, similarity


def run_model_svd(model_key, mol):
    if model_key == "Your Choice":
        return selected_product
    else:
        model = svd2svd_models[model_key]

        reactant = Chem.MolToSmiles(mol)
        reactant_FP = smiles_to_fp([reactant])
        reactant_svd = svd.transform(reactant_FP)
        predictions_svd = model.predict(reactant_svd)[0]
        predictions_FP = svd.inverse_transform([predictions_svd])
        if len(np.unique(predictions_FP)) != 2:
            predictions_FP = np.digitize(predictions_FP, bins=[0.5, 3])

        predicted_product = compare_FP_to_products_MCC(predictions_FP[0], possible_prods(Chem.MolFromSmiles(reactant)))[
            2]
        return predicted_product

intro_col1, intro_col2 = st.columns(2)

with intro_col1:
st.markdown("This app predicts the most likely product of a halogenation reaction on an aromatic systems. "
            "The AI model generates all possible halogenation products, and the predicted Morgan Fingerprint, "
            "then it matches the product to the most similar FP (Selected product)."
            "Below you can see a general scheme of the workflow: ")

st.image("workflow.png", use_column_width=True)

st.markdown(
    "This model was generated using a data-only approach, i.e. no prior knowledge of the reaction mechanism was used. "
    "To see it working, input the SMILES of the molecule you are interested in, then run the model.")

#st.markdown("It also give a confidence score for the prediction, which is based on the predicted FP similarity "
#            "and the similarity distribution within the range of possible products."
#            "its distribution among different possible products")

st.markdown("Disclaimer: No data is stored in this version of the app.")
st.markdown("----------------------------------------------------------------")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.markdown('Enter the molecule:')
    input_type = st.radio("Select your input method:", ("Draw it", "SMILES", "SDF file"))

    if input_type == "Draw it":
        smiles_string = st_ketcher("CC1=CC=CC2=C1C=CC=C2")

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

    if mol is  None:
        st.text('Invalid SMILES string')

    show_products_bar = False

    # show_products = st.button("Show possible products")

with col2:
    # col11, col22 = st.columns(2)

    # with col11:
    # Show possible products
    st.markdown("#### Predicted chlorination product")


    nb_halogens = st.select_slider("Number of halogenations", options=list(range(1, count_halogenations(mol)+1)), value=1)
    st.write("Number of halogenations: ", nb_halogens)
    products = []
    products_image = []


    potential_products = [Chem.MolToSmiles(mol)]
    products_list = [mol]

    for i in range(nb_halogens):
        products = []
        for potential_product in potential_products:
            for new_possible_prods in possible_prods(Chem.MolFromSmiles(potential_product)):
                #st.write(new_possible_prods)
                products.append(new_possible_prods)
        potential_products = products
        #Here!
        products_list.append(Chem.MolFromSmiles(run_model_fp("Ridge", products_list[i])[0]))

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



    # st.text('Possible products:')



    predictions = {"ML prediction": predicted_product}
    # similarity = run_model_fp("Ridge", mol)[1]
    # selected_product = st.radio("Select the major product:", options=range(len(products)), on_change=None)
    st.image(Draw.MolToImage(predictions["ML prediction"]))
    st.markdown("Product SMILES: " + Chem.MolToSmiles(predicted_product))
    # st.markdown("Confidence score: " + str(round(1 + max(similarity), 4) - 1))

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


    #for potential_product in potential_products:
    #    products.append(Chem.MolFromSmiles(potential_product))
    #    products_image.append(Draw.MolToImage(Chem.MolFromSmiles(potential_product), size=(100, 100)))

    if len(potential_products) == 0:
        st.text('No possible products found')
    # else:
    #    st.image(products_image)


    run_models = st.button("Refresh model output")


rxn_string = Chem.MolToSmiles(mol)
for molecule in products_list[1:]:
    rxn_string += "." + Chem.MolToSmiles(molecule)

st.markdown("#### Reaction scheme")
rxn_schema = st.columns([0.5, 2*len(products_list), 0.5])[1]
with rxn_schema:
    st.image(Draw.MolsToImage(products_list, molsPerRow=len(products_list), subImgSize=(250, 250)), use_column_width=True)
# predictions = {"Your Choice": products[selected_product],
#               "Ridge": Chem.MolFromSmiles(run_model_svd("Ridge", mol)),
#               "Ridge_10": Chem.MolFromSmiles(run_model_svd("Ridge_10", mol)),
#               "Ridge_50": Chem.MolFromSmiles(run_model_svd("Ridge_50", mol)),
#               "Lasso": Chem.MolFromSmiles(run_model_svd("Lasso", mol)),
#               "Elastic Net": Chem.MolFromSmiles(run_model_svd("Elastic Net", mol)),
#               "BayesianRidge": Chem.MolFromSmiles(run_model_svd("BayesianRidge", mol)),
#               "DecisionTreeRegressor": Chem.MolFromSmiles(run_model_svd("DecisionTreeRegressor", mol)),
#               "ExtraTreeRegressor": Chem.MolFromSmiles(run_model_svd("ExtraTreeRegressor", mol)),
#               "XGBoostRegressor": Chem.MolFromSmiles(run_model_svd("XGBoostRegressor", mol)),
#               "AdaBoostRegressor": Chem.MolFromSmiles(run_model_svd("AdaBoostRegressor", mol))
#               }


# Model outputs!
st.markdown("----------------------------------------------------------------")

st.markdown("### Possible products list")

col1, col2, col3 = st.columns(3)

  ##Use here an empty list that grows inside the for loop




data = {'Text': [f'Product  {i+1}' for i in range(len(products))],
        'Product': products_image,
        'Models': model_selection
        #'Similarity': similarity
        }

df = pd.DataFrame(data)

for index, row in df.iterrows():
    # st.write(model_selection[index])
    if index % 3 == 0:
        with col1:
            st.write(row['Text'])
            st.image(row['Product'], use_column_width=False)
            st.write(row['Models'])
            #st.write("Confidence: ", row['Similarity'])
    elif index % 3 == 1:
        with col2:
            st.write(row['Text'])
            st.image(row['Product'], use_column_width=False)
            st.write(row['Models'])
            #st.write("Confidence: ", row['Similarity'])
    elif index % 3 == 2:
        with col3:
            st.write(row['Text'])
            st.image(row['Product'], use_column_width=False)
            st.write(row['Models'])
            #st.write("Confidence: ", row['Similarity'])
