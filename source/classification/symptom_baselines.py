"""
All functions for running baselines for symptom status prediction
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
from sklearn.dummy import DummyClassifier
import clinlp, spacy 
from unidecode import unidecode
import re 

def majority_baseline(X_train, y_train, X_test):
    """
    Trains and runs a majority baseline classifier. When no X_test is given, predicted labels are on X_train. 

    Parameters: 
    X_train (pd.Series): train text from notes with NoteID as index
    y_train (pd.Series): train labels with NoteID as index
    X_test (pd.Series): train text from notes with NoteID as index, can be None  

    Returns: 
    predicted_labels (dict[str, int]]): all predicted labels as a dictionary of: NoteID -> predicted label 
    """

    ##setup dictionary 
    predicted_labels = {}

    ## set majority classifier 
    majority_cls = DummyClassifier()
    majority_cls.fit(X_train, y_train)

    ## make predictions 
    if X_test is not None: 
        y_pred = majority_cls.predict(X_test) 
        #save prediction per noteID
        for noteID, prediction in zip(X_test.index, y_pred):
            predicted_labels[noteID] = int(prediction)
    else:
        y_pred = majority_cls.predict(X_train) 
        #save prediction per noteID
        for noteID, prediction in zip(X_train.index, y_pred):
            predicted_labels[noteID] = int(prediction)

    return predicted_labels

def keyword_baseline(X_train, cat, X_test):
    """
    Runs a keyword based classifier. When no X_test is given, predicted labels are on X_train. 

    Parameters: 
    X_train (pd.Series): train text from notes with NoteID as index
    cat (int): category which is being classified
    X_test (pd.Series): train text from notes with NoteID as index, can be None  

    Returns: 
    predicted_labels (dict[str, int]]): all predicted labels as a dictionary of: NoteID -> predicted label 
    """

    if X_test is not None:
        X = X_test
    else:
        X = X_train

    nlp = spacy.blank('clinlp')
    nlp.add_pipe("clinlp_sentencizer")

    ##setup dictionary 
    predicted_labels = {}

    for noteID, note in X.items():
        doc = nlp(note)
        predicted_labels[noteID] = 100 #no mention is the default label

        for item in doc.sents: #sentizer from clinlp to split into sentences 
            sent = unidecode((item.text).lower()) #normalize 
            label = match_keyword(sent, cat)
            if label is not None: #we take the first match 
                predicted_labels[noteID] = label
                break

    return predicted_labels

def match_keyword(sent, cat):
    """
    Looks for a match between a sentence and the keywords/patterns of a specific category. Depending on this it returns a predicted label.

    Parameters:
    sent (str): sentence to be matched
    cat (int): category which keywords/patterns should be used

    Returns: 
    label (int): the label corresponding to the specific match, or None if no match   
    """
    
    #All patterns for each category 
    lookup_dict = {
        0: {
            "general": [
                "depressieve", "stemming", "somber", "moedeloos", "hopeloos", "non-verbale", "verdrietig", "voelt zich slecht", "neerslachtig", "malaise", "treurig", "melancholiek", "dysthymie"
            ],
            "worsened": [
                "depressiever", "slechter", "moedelozer", "hopelozer", "treuriger", "verdrietiger", "somberder"
            ],
            "more": ["meer"],
            "improved": ["beter", "opgewekter", "positiever"],
            "less": ["minder"]
        },

        1: {
            "general": [
                "zelfdepreciatie", "(gebrek aan|lage|weinig|geen) zelfwaardering", "zelfverwijt", "(gebrek aan|weinig|geen) zelfrespect", "schuld", "straf", "faalangst", "(laag|weinig) zelfvertrouwen", "waardeloos", "inferieur", "onzeker"
            ],
            "worsened": [
                "schuldiger", "waardelozer", "(lager|minder|afname \w*) (zelfvertrouwen|zelfrespect|zelfwaardering)", "onzekerder"
            ],
            "more": ["meer"],
            "improved": [
                "zelfverzekerder", "zekerder", "beter zelfbeeld", "(meer|hoger|toename \w*) (zelfvertrouwen|zelfrespect|zelfwaardering)"
            ],
            "less": ["minder"]
        },

        2: {
            "general": [
                "leven", "sterven", "zelfmoord", "suicid", "doodswens", "euthanasie", "TS"
            ],
            "worsened": ["suicidaler"],
            "more": ["meer", "recent"],
            "improved": ["leefwens"],
            "less": ["minder"]
        },

        3: {
            "general": [
                "slecht slapen", "(niet|slecht) doorslapen", "wakker liggen", "insomni"
            ],
            "worsened": [
                "slechter slapen", "(minder|slechter) doorslapen"
            ],
            "more": ["meer"],
            "improved": [
                "(meer|beter) doorslapen", "beter slapen"
            ],
            "less": ["minder"]
        },

        4: {
            "general": [
                "anhedonie", "(geen|gebrek aan|weinig) interesse", "(geen|gebrek aan|weinig) motivatie", "geen activiteit", "werkt niet", "weinig plezier", "gedemotiveerd"
            ],
            "worsened": [
                "minder plezier", "gedemotiveerder", "(minder|afname \w*) (interesse|motivatie|activiteit|actief)", "werkt minder"
            ],
            "more": ["meer"],
            "improved": [
                "gemotiveerder", "actiever", "geïnteresseerder", "(toename \w*|meer) (interesse|motivatie)", "werkt meer"
            ],
            "less": ["minder"]
        },

        5: {
            "general": [
                "vertraging", "vertraagd", "langzaam", "vlak", "verminderd modulerend affect", "trage", "traag"
            ],
            "worsened": ["trager", "vlakker", "langzamer"],
            "more": ["meer"],
            "improved": ["sneller", "levendiger"],
            "less": ["minder"]
        },

        6: {
            "general": [
                "opwinding", "agitatie", "lichamelijke onrust", "rusteloosheid", "(moeite met|niet) stilzitten", "hyperactiviteit", "gejaagd"
            ],
            "worsened": [
                "onrustiger", "gejaagder", "rustelozer", "lichamelijk onrustiger"
            ],
            "more": ["meer"],
            "improved": ["rustiger", "kalmer"],
            "less": ["minder"]
        },

        7: {
            "general": [
                "angst", "bezorgdheid", "bang", "onveiligheid", "bedreiging", "paniek", "gespannenheid", "piekeren", "prikkelbaarheid", "vrees", "hypervigilantie", "nervositeit", "zenuwachtig"
            ],
            "worsened": [
                "banger", "paniekeriger", "prikkelbaarder", "angstiger", "onveiliger", "gespannener", "zenuwachtiger"
            ],
            "more": ["meer"],
            "improved": ["kalmer", "meer ontspannen"],
            "less": ["minder"]
        },

        8: {
            "general": [
                "gastro-intestina", "(weinig|geen|gebrek aan) (eetlust|voedselinname|smaak|energie)", "verstopping", "krampen", "transpireren", "beven", "hyperventilatie", "droge mond", "pijn", "vermoeid", "uitputting", "bleke indruk", "magere indruk", "dyspepsie", "misselijk"
            ],
            "worsened": [
                "vermoeider", "(afname \w*|minder) (eetlust|voedselinname|smaak|energie)", "misselijker"
            ],
            "more": ["meer"],
            "improved": [
                "(toename \w*|meer) (eetlust|voedselinname|smaak|energie)"
            ],
            "less": ["minder"]
        },

        9: {
            "general": [
                "hypochondrie", "hypochondrisch", "angst voor ziekte", "bezorgd om gezondheid", "lichamelijke gewaarwording", "lichamelijke symptomen", "ernstige ziekte", "waangedacht", "overbezorgd"
            ],
            "worsened": ["overbezorgder", "hypochondrischer"],
            "more": ["meer"],
            "improved": [],
            "less": ["minder"]
        },

        10: {
            "general": [
                "(gebrek aan|verminderd|afname \w*) inzicht", "ontkenning", "ongerelateerde factoren", "inzichtloos"
            ],
            "worsened": ["minder inzicht"],
            "more": ["meer"],
            "improved": ["meer inzicht"],
            "less": ["minder"]
        },

        11: {
            "general": ["gewichtsverlies"],
            "worsened": [
                "lichter", "afgenomen gewicht", "afgevallen", "weegt minder", "gewichtsverlies"
            ],
            "more": ["meer"],
            "improved": ["aangekomen", "weegt meer"],
            "less": ["minder"]
        }
    }

    # the patterns for the specific category 
    cat_dict = lookup_dict[cat]
    
    #First try to match with the most specific categories 
    for p in cat_dict['improved']:
        match = re.search(p, sent)
        if match:
            return 1 

    for p in cat_dict['worsened']:
        match = re.search(p, sent)
        if match:
            return -1

    for p in cat_dict['general']:
        match = re.search(p, sent)
        if match: #if there is match for 'general', check for 'more' or 'less' modifier 
            for p in cat_dict['less']:
                match = re.search(p, sent)
                if match:
                    return 1
            for p in cat_dict['more']:
                match = re.search(p, sent)
                if match:
                    return -1 
            return 0 #no modifiers 
        
    return None
