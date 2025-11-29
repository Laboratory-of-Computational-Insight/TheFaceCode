import gc
from builtins import enumerate
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import statsmodels.api as sm


from sklearn.preprocessing import StandardScaler

from ResidualMaskingNetwork.rmn import RMN
from VGG.wrapper_vgg import WrapVGG
from analyze.lower_dim import pca_lower_dim, pca_get_dim, cumsum_index

from data_loader.fer2013_data_loader import Emotions, seven_to_three_conversion

from data_load import disfa, fer2013
from emonet.emonet.models import EmoNet
from models.net import Net
from models.net_smaller import Net as NetSmaller
from objects.context import get_context
from objects.label_types import Attribute
from utils.cs_plot.cs_plot import plot_embedding, remove_outlayers

from utils.model import create_model
from collections import defaultdict

from sklearn.metrics import silhouette_score
import torch


layer_attribute_param = {}



context = get_context()

def get_all_emotions_elements():
    emotions_all = [e for e in Emotions]
    return emotions_all

def get_dummy_conversion():
    conversion_dummy = {e.value: e.value for e in Emotions}
    conversion_dummy[7] = 1
    return conversion_dummy



def create_model3_no_p3_dim(context, i=0):
    file = "./models/submitted/em3_no_p3_dimensional/epoch99_last.pth"

    print(file)
    model3, last_epoch3 = create_model(context, 3, Net, file, context.device)
    model3.eval()
    # model3.reset_first_conv()

    elements_corolation3 = [
        ("batch_norm1", 4, model3.batch_norm1),
        ("flatten", 4, model3.flatten),
        ("dropout_4", 4, model3.dropout_4),
        # ("softmax", 2, None)
        ]
    emotions_3 = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion3 = seven_to_three_conversion
    return "3_emotions_no_p3_dim", model3, model3, emotions_3, conversion3, elements_corolation3



def create_model3_yes_p3_iden(context, i=0):
    file = "./models/submitted/em3_yes_p3_iden/epoch99_last.pth"

    print(file)
    model3, last_epoch3 = create_model(context, 3, Net, file, context.device)
    model3.eval()
    # model3.reset_first_conv()

    elements_corolation3 = [
        ("batch_norm1", 4, model3.batch_norm1),
        ("flatten", 4, model3.flatten),
        ("dropout_4", 4, model3.dropout_4),
        # ("softmax", 2, None)
        ]
    emotions_3 = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion3 = seven_to_three_conversion
    return "3_emotions_yes_p3_iden", model3, model3, emotions_3, conversion3, elements_corolation3



def create_model3_no_p1_iden(context, i=0):
    file = "./models/submitted/em3_no_p1_iden/epoch99_last.pth"

    print(file)
    model3, last_epoch3 = create_model(context, 3, Net, file, context.device)
    model3.eval()
    # model3.reset_first_conv()

    elements_corolation3 = [
        ("batch_norm1", 4, model3.batch_norm1),
        ("flatten", 4, model3.flatten),
        ("dropout_4", 4, model3.dropout_4),
        # ("softmax", 2, None)
        ]
    emotions_3 = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion3 = seven_to_three_conversion
    return "3_emotions_no_p1_iden", model3, model3, emotions_3, conversion3, elements_corolation3



def create_model3_no_raw_iden(context, i=0):
    file = "./models/submitted/em3_no_raw_iden/epoch99_last.pth"

    print(file)
    model3, last_epoch3 = create_model(context, 3, Net, file, context.device)
    model3.eval()
    # model3.reset_first_conv()

    elements_corolation3 = [
        ("batch_norm1", 4, model3.batch_norm1),
        ("flatten", 4, model3.flatten),
        ("dropout_4", 4, model3.dropout_4),
        # ("softmax", 2, None)
        ]
    emotions_3 = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion3 = seven_to_three_conversion
    return "3_emotions_no_raw_iden", model3, model3, emotions_3, conversion3, elements_corolation3



def create_model3(context, i=0):
    file = "./models/submitted/em3_normal/epoch99_last.pth"

    print(file)
    model3, last_epoch3 = create_model(context, 3, Net, file, context.device)
    model3.eval()
    # model3.reset_first_conv()

    elements_corolation3 = [
        ("batch_norm1", 4, model3.batch_norm1),
        ("flatten", 4, model3.flatten),
        ("dropout_4", 4, model3.dropout_4),
        # ("softmax", 2, None)
        ]
    emotions_3 = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion3 = seven_to_three_conversion
    return "3_emotions", model3, model3, emotions_3, conversion3, elements_corolation3


def create_model3_smaller(context, i=0):
    file = f"./models/submitted/em3_smaller/epoch99_last.pth"
    print(file)
    model3, last_epoch3 = create_model(context, 3, NetSmaller, file, context.device)
    model3.eval()
    # model3.reset_first_conv()

    elements_corolation3 = [
        ("batch_norm1", 4, model3.batch_norm1),
        ("flatten", 4, model3.flatten),
        ("dropout_4", 4, model3.dropout_4),
        # ("softmax", 2, None)
        ]
    emotions_3 = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion3 = seven_to_three_conversion
    return "3_emotions_smaller", model3, model3, emotions_3, conversion3, elements_corolation3



def create_model7_60(context):
    model7_60, last_epoch7_60 = create_model(context, 7, Net, "./models/submitted/em7_normal/epoch99_last.pth", context.device)
    model7_60.eval()
    emotions_all = get_all_emotions_elements()
    conversion_dummy = get_dummy_conversion()
    elements_corolation7_60 = [
        ("batch_norm1", 4, model7_60.batch_norm1),
        ("flatten", 4, model7_60.flatten),
        ("dropout_4", 2, model7_60.dropout_4),
        # ("softmax", 2, None)
    ]

    return "7_emotions", model7_60, model7_60, emotions_all, conversion_dummy, elements_corolation7_60


def create_model_rmn(context):
    rmn = RMN()
    elements_corolation_rmn = [
        # ("first", 4, None),
        # ("second", 4, None),
        ("third", 2, None),
        # ("forth", 2, None)
    ]
    emotions_all = get_all_emotions_elements()
    conversion_dummy = get_dummy_conversion()
    return "rmn", rmn, rmn.torch_detect_emotion_for_single_frame, emotions_all, conversion_dummy, elements_corolation_rmn

def create_model_emofan(context):
    n_expression = 8
    state_dict_path = Path(__file__).parent.joinpath('./emonet/pretrained', f'emonet_{n_expression}.pth')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    net = EmoNet(n_expression=n_expression).to("cuda")
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    elements_corolation_emofan = [
        # ("bn1", 4, None),
        # ("emo_feat", 4, None),
        ("emo_fc_2", 2, None),
        # ("final", 2, None)
    ]
    emotions_all = get_all_emotions_elements()
    conversion_dummy = get_dummy_conversion()
    return "emofan", net,net.forward, emotions_all, conversion_dummy, elements_corolation_emofan


def create_model_vgg(context):
    vgg = WrapVGG("cuda")
    elements_corolation_vgg = [
        ("first", 4, None),
        # ("second", 4, None),
        ("third", 2, None),
        # ("forth", 2, None)
    ]
    emotions_all = get_all_emotions_elements()
    conversion_dummy = get_dummy_conversion()
    return "vgg", vgg, vgg, emotions_all, conversion_dummy, elements_corolation_vgg



datas_labels = [
    ["disfa_emofan", *disfa()],
    ["fer2013_emofan", *fer2013()],
]

models = [
    create_model3(context,1), # ["3_emotions", model3, model3, emotions_3, conversion3, elements_corolation3, elements_gradscan3],
    create_model3_smaller(context,1), # ["3_emotions", model3, model3, emotions_3, conversion3, elements_corolation3, elements_gradscan3],
    create_model3_no_raw_iden(context,1), # ["3_emotions", model3, model3, emotions_3, conversion3, elements_corolation3, elements_gradscan3],
    create_model3_no_p1_iden(context,1), # ["3_emotions", model3, model3, emotions_3, conversion3, elements_corolation3, elements_gradscan3],
    create_model3_yes_p3_iden(context,1), # ["3_emotions", model3, model3, emotions_3, conversion3, elements_corolation3, elements_gradscan3],
    create_model3_no_p3_dim(context,1), # ["3_emotions", model3, model3, emotions_3, conversion3, elements_corolation3, elements_gradscan3],
    create_model7_60(context), # ["7_emotions_60", model7_60, model7_60, emotions_all, conversion_dummy, elements_corolation7_60, elements_gradscan7_60],
    create_model_rmn(context), # ["rmn", rmn, rmn.torch_detect_emotion_for_single_frame, emotions_all, conversion_dummy, elements_corolation_rmn, elements_gradscan_rmn],
    create_model_emofan(context), # ["emofan", net,net.forward2, emotions_all, conversion_dummy, elements_corolation_emofan, elements_gradscan_emofan],
    create_model_vgg(context) # ["vgg", vgg, vgg, emotions_all, conversion_dummy, elements_corolation_vgg, elements_gradscan_vgg]
    ]


silhouette_score_results = {} # model-data-layer-dim_reduction: score
regression_results = {}

print("starting")
for data_name, data_orig ,labels_orig in datas_labels:
    print("\t",data_name)

    base_name = data_name.split("_")[0]
    images_for_video = {}

    if "fer" in data_name:
        limit = 1000
    else:
        limit = 13_000

    MAX_PER_ID = 1000
    # Count how many samples we keep per identity and build index map
    id_counts = defaultdict(int)
    index_map = []
    for idx, label in labels_orig.items():
        id_temp = label[0]  # identity is labels_orig[idx][0]
        id_counts[id_temp] += 1
        if id_counts[id_temp] < MAX_PER_ID:
            index_map.append(idx)
    # Subset data
    data_orig = data_orig[index_map]
    # Rebuild labels_orig with contiguous indices
    labels_orig = {new_i: labels_orig[old_i] for new_i, old_i in enumerate(index_map)}
    # Simple index tensor
    rand_index_pre_change = torch.arange(len(data_orig))

    for model_name, model, model_f, emotions, conversion, elements_corolation in models:
        print("\t\t", model_name)

        # emofan model is too intense. either limit the data or save the interemediate results seperatly at each iteration (run this multiple times)
        if model_name == "emofan":
            # limit = 1000
            rand_index = rand_index_pre_change[:limit]
        else:
            rand_index = rand_index_pre_change

        model.history.clear()
        gc.collect()

        # Shuffle / reorder data and labels according to rand_index
        data = data_orig[rand_index]
        labels = {new_i: labels_orig[i.item()] for new_i, i in enumerate(rand_index)}
        # Keep only samples whose emotion is in `emotions`
        valid_indices = [
            i for i in range(len(data))
            if labels[i][Attribute.EMOTION] in emotions
        ]
        # Filter data
        data = data[valid_indices]
        # Prepare new labels + emotion distribution
        labels_temp = {}
        emo_dist = {e: 0 for e in emotions}
        for i in valid_indices:
            label = list(labels[i])
            emo = label[Attribute.EMOTION]
            emo_dist[emo] += 1
            label[Attribute.EMOTION] = conversion[emo]  # remap emotion
            labels_temp[len(labels_temp)] = tuple(label)
        labels = labels_temp
        if hasattr(model, "save_history_flag"):
            model.save_history_flag(True, elements=[e[0] for e in elements_corolation])

        data = data.to(context.device)
        predictions = []

        delete = []
        for i in range(len(data)):
            instance = data[i:i + 1]
            prediction = model_f(instance)
            try:
                predictions.append(
                    prediction.argmax().item()
                )
            except:
                #emofan returns at a slightly different format. perhaps the best fix is to change forward2
                disfa_conversion = {  # emo_net: our_model
                    0: 6,  # neutral
                    1: 3,  # Happy
                    2: 4,  # Sad
                    3: 5,  # Surprise
                    4: 2,  # Fear
                    5: 1,  # Disgust
                    6: 0,  # Anger
                    7: 1,  # Contempt
                }
                predictions.append(disfa_conversion[prediction["expression"].argmax().item()])
        for i in reversed(delete):
            del data[i]
            del labels[i]

        print("\t\t\t", "accuracy", (np.array([l[Attribute.EMOTION] for l in labels.values()]) == np.array(predictions)).mean())

        for element, percent_index_80, sub_model in elements_corolation:
            print("\t\t\t", element)
            intermidate_predictions = torch.cat([h[element] for h in model.history], dim=0)

            for h in model.history:
                del h[element]

            for attribute in [
                # Attribute.EMOTION,
                Attribute.VALENCE,
                # Attribute.AROUSAL,
                # Attribute.IDENTITY
            ]:
                gc.collect()
                print("\t\t\t\t", attribute)

                colors = [labels[i][attribute] for i in range(len(intermidate_predictions))]
                if attribute == Attribute.EMOTION: # perception
                    colors = [predictions[i] for i in range(len(intermidate_predictions))]

                flattened = intermidate_predictions.reshape(len(intermidate_predictions), -1)
                h_index, h_value, l_index, l_value = pca_get_dim(flattened, 0.85, n_components=128)

                h_embedings = pca_lower_dim(intermidate_predictions, n_components=h_index)

                h_embedings, colors_h, preds_h, _ = remove_outlayers(h_embedings, colors, predictions)
                h_embedings, colors_h, preds_h, _ = remove_outlayers(h_embedings, colors_h, preds_h)

                sil_h = silhouette_score(h_embedings, colors_h) if attribute in [Attribute.EMOTION, Attribute.IDENTITY] else -1 #only calc silhouette if the coloring is discrete (Emotion or identity)
                print("\t\t\t\t\tsil", sil_h)

                if attribute == Attribute.IDENTITY and element == "dropout_4":
                    pca = PCA(n_components=min(intermidate_predictions.shape))
                    pca.fit(intermidate_predictions.to("cpu").reshape(len(intermidate_predictions), -1))
                    index, value, minus_one_index, minus_one_value = cumsum_index(0.85, pca.explained_variance_ratio_)
                    lower = (intermidate_predictions - torch.tensor(pca.mean_).to(intermidate_predictions.device)) @ torch.tensor(pca.components_[index:].T).to(intermidate_predictions.device)
                    lower, colors_h, preds_h, _ = remove_outlayers(lower, colors, predictions)
                    lower, colors_h, preds_h, _ = remove_outlayers(lower, colors_h, preds_h)

                    sil_of_iden_inverse = silhouette_score(lower, colors_h)
                    print("\t\t\t\t\t inverse sil", sil_of_iden_inverse)

                layer_attribute_param[element] = layer_attribute_param.get(element, {})
                layer_attribute_param[element][attribute] = layer_attribute_param[element].get(attribute, {})
                layer_attribute_param[element][attribute]["silhouette"] = layer_attribute_param[element][attribute].get("silhouette", [])
                layer_attribute_param[element][attribute]["silhouette"].append(sil_h)


                flattened = intermidate_predictions.reshape(len(intermidate_predictions), -1)
                pca10 = pca_lower_dim(flattened, n_components=10)

                x = pca10[:,0]
                y = pca10[:,1]
                xys = np.array([x, y]).T
                c = colors
                scaler = StandardScaler()
                axis = scaler.fit_transform(xys)
                reg = LinearRegression().fit(axis.reshape(-1, 2), np.array(colors).reshape(-1, 1))
                score = reg.score(axis.reshape(-1, 2), np.array(colors).reshape(-1, 1))
                print("\t\t\t\t\t\t score", score, )
                f_stats, p_values = f_regression(axis.reshape(-1, 2), np.array(colors))
                print("\t\t\t\t\t\t", "coefs", reg.coef_, reg.intercept_, "stats", f_stats, "pvalue", p_values, "score", score)



                layer_attribute_param[element] = layer_attribute_param.get(element, {})
                layer_attribute_param[element][attribute] = layer_attribute_param[element].get(attribute, {})
                for am in ["coefs_x", "coefs_y", "intercept", "score", "fstats_f", "fstats_p", "square", "square_adj", "pvalx", "pvaly"]:
                    layer_attribute_param[element][attribute][am] = layer_attribute_param[element][attribute].get(am, [])

                mod = sm.OLS(colors, axis.reshape(-1, 2), fit_intercept=True)
                fii = mod.fit()
                # print(fii.summary())
                p_values = fii.summary2().tables[1]['P>|t|']
                layer_attribute_param[element][attribute]["coefs_x"].append(reg.coef_[0,0])
                layer_attribute_param[element][attribute]["coefs_y"].append(reg.coef_[0,1])
                layer_attribute_param[element][attribute]["intercept"].append(reg.intercept_[0])
                layer_attribute_param[element][attribute]["score"].append(score)
                layer_attribute_param[element][attribute]["fstats_f"].append(fii.fvalue)
                layer_attribute_param[element][attribute]["fstats_p"].append(fii.f_pvalue)
                layer_attribute_param[element][attribute]["square"].append(fii.rsquared)
                layer_attribute_param[element][attribute]["square_adj"].append(fii.rsquared_adj)
                layer_attribute_param[element][attribute]["pvalx"].append(p_values["x1"])
                layer_attribute_param[element][attribute]["pvaly"].append(p_values["x2"])
                print("\t\t\t\t\t\t", layer_attribute_param)

                lower = pca_lower_dim(flattened, n_components=2)
                is_dimensional = attribute in (Attribute.VALENCE, Attribute.AROUSAL)
                new_colors = [
                    colors[d] if is_dimensional else str(colors[d])
                    for d in range(len(data))
                ]
                new_lower = np.asarray(lower)
                new_predictions = [str(p) for p in predictions]

                plot_embedding(new_lower, title=f"embeddings {data_name} {model_name} {element} {attribute.name} ;sil_h {sil_h:.2f}", colors=new_colors)

        model.history.clear()
        gc.collect()


print("DONE:\n",layer_attribute_param)
