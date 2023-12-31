
from collections import OrderedDict

FITZ17K_CLASSES = OrderedDict({
    "0": "drug induced pigmentary changes",
    "1": "photodermatoses",
    "2": "dermatofibroma",
    "3": "psoriasis",
    "4": "kaposi sarcoma",
    "5": "neutrophilic dermatoses",
    "6": "granuloma annulare",
    "7": "nematode infection",
    "8": "allergic contact dermatitis",
    "9": "necrobiosis lipoidica",
    "10": "hidradenitis",
    "11": "melanoma",
    "12": "acne vulgaris",
    "13": "sarcoidosis",
    "14": "xeroderma pigmentosum",
    "15": "actinic keratosis",
    "16": "scleroderma",
    "17": "syringoma",
    "18": "folliculitis",
    "19": "pityriasis lichenoides chronica",
    "20": "porphyria",
    "21": "dyshidrotic eczema",
    "22": "seborrheic dermatitis",
    "23": "prurigo nodularis",
    "24": "acne",
    "25": "neurofibromatosis",
    "26": "eczema",
    "27": "pediculosis lids",
    "28": "basal cell carcinoma",
    "29": "pityriasis rubra pilaris",
    "30": "pityriasis rosea",
    "31": "livedo reticularis",
    "32": "stevens johnson syndrome",
    "33": "erythema multiforme",
    "34": "acrodermatitis enteropathica",
    "35": "epidermolysis bullosa",
    "36": "dermatomyositis",
    "37": "urticaria",
    "38": "basal cell carcinoma morpheiform",
    "39": "vitiligo",
    "40": "erythema nodosum",
    "41": "lupus erythematosus",
    "42": "lichen planus",
    "43": "sun damaged skin",
    "44": "drug eruption",
    "45": "scabies",
    "46": "cheilitis",
    "47": "urticaria pigmentosa",
    "48": "behcets disease",
    "49": "nevocytic nevus",
    "50": "mycosis fungoides",
    "51": "superficial spreading melanoma ssm",
    "52": "porokeratosis of mibelli",
    "53": "juvenile xanthogranuloma",
    "54": "milia",
    "55": "granuloma pyogenic",
    "56": "papilomatosis confluentes and reticulate",
    "57": "neurotic excoriations",
    "58": "epidermal nevus",
    "59": "naevus comedonicus",
    "60": "erythema annulare centrifigum",
    "61": "pilar cyst",
    "62": "pustular psoriasis",
    "63": "ichthyosis vulgaris",
    "64": "lyme disease",
    "65": "striae",
    "66": "rhinophyma",
    "67": "calcinosis cutis",
    "68": "stasis edema",
    "69": "neurodermatitis",
    "70": "congenital nevus",
    "71": "squamous cell carcinoma",
    "72": "mucinosis",
    "73": "keratosis pilaris",
    "74": "keloid",
    "75": "tuberous sclerosis",
    "76": "acquired autoimmune bullous diseaseherpes gestationis",
    "77": "fixed eruptions",
    "78": "lentigo maligna",
    "79": "lichen simplex",
    "80": "dariers disease",
    "81": "lymphangioma",
    "82": "pilomatricoma",
    "83": "lupus subacute",
    "84": "perioral dermatitis",
    "85": "disseminated actinic porokeratosis",
    "86": "erythema elevatum diutinum",
    "87": "halo nevus",
    "88": "aplasia cutis",
    "89": "incontinentia pigmenti",
    "90": "tick bite",
    "91": "fordyce spots",
    "92": "telangiectases",
    "93": "solid cystic basal cell carcinoma",
    "94": "paronychia",
    "95": "becker nevus",
    "96": "pyogenic granuloma",
    "97": "langerhans cell histiocytosis",
    "98": "port wine stain",
    "99": "malignant melanoma",
    "100": "factitial dermatitis",
    "101": "xanthomas",
    "102": "nevus sebaceous of jadassohn",
    "103": "hailey hailey disease",
    "104": "scleromyxedema",
    "105": "porokeratosis actinic",
    "106": "rosacea",
    "107": "acanthosis nigricans",
    "108": "myiasis",
    "109": "seborrheic keratosis",
    "110": "mucous cyst",
    "111": "lichen amyloidosis",
    "112": "ehlers danlos syndrome",
    "113": "tungiasis",
})

generic_classes = {}
for i in range(0, 114):
    generic_classes[str(i)] = "skin disease"

FITZ17K_GENERIC_CLASSES = OrderedDict(generic_classes)