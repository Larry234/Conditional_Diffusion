import torch
from torch import nn
import torch.nn.functional as F

from .encoder import *


class CCIPModel(nn.Module):
    def __init__(
        self,
        num_atr,
        num_obj,
        temperature=1,
        image_embedding=2048,
        class_embedding=512,
        projection_dim=256,
    ):
        super().__init__()
        
        self.num_atr = num_atr
        self.num_obj = num_obj
        self.class_emb_dim = class_embedding
        self.temperature = temperature
        self.image_encoder = ImageEncoder()
        self.class_encoder = ClassEncoder(num_atr + 1, num_obj + 1, emb_dim=class_embedding)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.class_projection = ProjectionHead(embedding_dim=class_embedding * 2, projection_dim=projection_dim)
        

    def forward(self, image, atr, obj):
        # Getting Image and Text Features
        image_features = self.image_encoder(image)
        class_features = self.class_encoder(atr=atr, obj=obj)
        
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        class_embeddings = self.class_projection(class_features)
        
        # Calculating the Loss
        logits = (class_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = class_embeddings @ class_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class CMLIPModel(nn.Module):
    def __init__(
        self,
        num_size,
        num_atr,
        num_obj,
        temperature=1,
        image_embedding=2048,
        class_embedding=512,
        projection_dim=256,
    ):
        super().__init__()
        self.num_size = num_size
        self.num_atr = num_atr
        self.num_obj = num_obj
        self.class_emb_dim = class_embedding
        self.temperature = temperature
        self.image_encoder = ImageEncoder()
        self.class_encoder = TripleClassEncoder(num_size + 1, num_atr + 1, num_obj + 1, emb_dim=class_embedding)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.class_projection = ProjectionHead(embedding_dim=class_embedding * 3, projection_dim=projection_dim)
        
    def encode_class(self, size, atr, obj):
        emb = self.class_encoder(size=size, atr=atr, obj=obj)
        return self.class_projection(emb)
    
    def encode_image(self, image):
        emb = self.image_encoder(image)
        return self.image_projection(emb)

    def forward(self, image, size, atr, obj):
        # Getting Image and Text Features
        image_features = self.image_encoder(image)
        class_features = self.class_encoder(size=size, atr=atr, obj=obj)
        
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        class_embeddings = self.class_projection(class_features)
        
        # Calculating the Loss
        logits = (class_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = class_embeddings @ class_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")