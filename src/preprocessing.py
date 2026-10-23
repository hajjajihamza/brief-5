from sklearn.preprocessing import LabelEncoder

def applay_encoding(list):
    encoder = LabelEncoder()
    return encoder.fit_transform(list), encoder