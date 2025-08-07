from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

def train_ctgan_model(train_data, epochs=1000):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)
    model = CTGANSynthesizer(epochs=epochs, verbose=True, metadata=metadata)
    model.fit(train_data)
    synthetic_data = model.sample(len(train_data))
    return synthetic_data


def train_tvae_model(train_data, epochs=1000):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)
    model = TVAESynthesizer(
        embedding_dim=128, compress_dims=(128, 64), decompress_dims=(64, 128),
        epochs=epochs, verbose=True, metadata=metadata
    )
    model.fit(train_data)
    synthetic_data = model.sample(len(train_data))
    return synthetic_data
