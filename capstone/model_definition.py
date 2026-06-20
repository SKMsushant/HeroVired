import tensorflow as tf
import numpy as np

# ==========================================
# Custom Blocks Definition
# ==========================================
class Cbs_Block(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides=1,padding='same',**kwargs):

        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv=tf.keras.layers.Conv2D(filters,kernel_size,strides=strides,padding=padding,use_bias=False)
        self.bn=tf.keras.layers.BatchNormalization()
        self.act=tf.keras.layers.Activation('silu')

    def call(self,x,training=False):
        return self.act(self.bn(self.conv(x),training=training))

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config

    
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self,filters,shortcut=True,**kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.cbs1=Cbs_Block(filters,kernel_size=3)
        self.cbs2=Cbs_Block(filters,kernel_size=3)
        self.shortcut=shortcut

    def call(self,x,training=False):
        out=self.cbs2(self.cbs1(x,training=training),training=training)
        return x+out if self.shortcut else out    

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "shortcut": self.shortcut
        })
        return config

    
class C2f_Block(tf.keras.layers.Layer):
    def __init__(self,filters,num_bottleneck=1,shortcut=True,**kwargs):

        super().__init__(**kwargs)
        self.filters = filters
        self.num_bottleneck = num_bottleneck
        self.shortcut = shortcut
        self.filter_branch=filters//2
        self.c2f_cbs_top=Cbs_Block(filters=2*self.filter_branch,kernel_size=1)
        self.c2f_cbs_bottom=Cbs_Block(filters=filters,kernel_size=1)
        self.bottlenecks=[BottleNeck(self.filter_branch,shortcut=True) for _ in range(num_bottleneck)]

    def call(self,x,training=False):
        split_candidate=self.c2f_cbs_top(x,training=training)
        y=list(tf.split(split_candidate,num_or_size_splits=2,axis=-1))

        for m in self.bottlenecks:
            y.append(m(y[-1],training=training))

        merged=tf.keras.layers.Concatenate(axis=-1)(y)

        return self.c2f_cbs_bottom(merged,training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "num_bottleneck": self.num_bottleneck,
            "shortcut": self.shortcut
        })
        return config


class Sppf_Block(tf.keras.layers.Layer):
    def __init__(self,filters,pool_size=5,**kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.pool_size = pool_size
        self.filter_branch=filters//2
        self.sppf_cbs_top=Cbs_Block(self.filter_branch,1)
        self.sppf_cbs_bottom=Cbs_Block(filters,1)
        self.pool=tf.keras.layers.MaxPooling2D(pool_size=pool_size,strides=1,padding='same')


    def call(self,x,training=False):
        x_sppf_cbs_top=self.sppf_cbs_top(x,training=training)
        y1=self.pool(x_sppf_cbs_top)
        y2=self.pool(y1)
        y3=self.pool(y2)

        merged=tf.keras.layers.Concatenate(axis=-1)([x_sppf_cbs_top,y1,y2,y3])

        return self.sppf_cbs_bottom(merged,training=training)


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "pool_size": self.pool_size
        })
        return config
    



class Static_Positional_Embedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

        sentence_positional_vector_list = []
        for word_position in range(self.seq_len):
            word_position_vector = []
            for i in range(int((self.d_model / 2))):
                y_sin = np.sin(word_position / 10000 ** ((2 * i) / self.d_model))
                y_cos = np.cos(word_position / 10000 ** ((2 * i) / self.d_model))
                word_position_vector.append(y_sin)
                word_position_vector.append(y_cos)
            sentence_positional_vector_list.append(word_position_vector)

        encoding_matrix = np.array(sentence_positional_vector_list)
        self.positional_encoding = tf.constant(encoding_matrix, dtype=tf.float32)[None, ...]

    def call(self, x):
        return x + self.positional_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model
        })
        return config
    
class Transformer_Encoder(tf.keras.layers.Layer):

    def __init__(self,num_heads,d_model,ff_dim,dropout=0.1,kernel_regularizer=None,**kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer

        self.mha=tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=d_model,value_dim=d_model)
        self.ffn=tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim,activation='relu',kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dense(d_model,kernel_regularizer=kernel_regularizer)
        ])
        self.layernorm_mha=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_ffn=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_mha=tf.keras.layers.Dropout(dropout)
        self.dropout_ffn=tf.keras.layers.Dropout(dropout)


    def call(self,x,training=False):
        #Multi-Head Attention and residual connection
        attn_output=self.mha(x,x,x,training=training)
        attn_output=self.dropout_mha(attn_output,training=training)
        out1=self.layernorm_mha(x+attn_output)

        #Feed Forward network and residual connection
        ffn_output=self.ffn(out1,training=training)
        ffn_output=self.dropout_ffn(ffn_output,training=training)
        return self.layernorm_ffn(out1+ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


# ==========================================
# Model Reconstruction Function
# ==========================================

def build_optimal_model(num_classes, img_size=224, hps_dict=None):
    """
    Rebuilds the hybrid YOLOv8 + Transformer model using the optimal 
    hyperparameters dictionary (loaded from best_hps.json).
    """
    if hps_dict is None:
        hps_dict = {
            "reg_type": "None",
            "lr": 1e-4
        }

    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3), name='input_layer')

    # --- Data Augmentation (Identity during inference) ---
    x = tf.keras.layers.RandomFlip('horizontal_and_vertical', name='random_flip')(inputs)
    x = tf.keras.layers.RandomRotation(0.2, name='random_rotation')(x)
    x = tf.keras.layers.RandomContrast(0.15, name='random_contrast')(x)
    x = tf.keras.layers.RandomZoom(0.1, name='random_zoom')(x)

    # --- CSPDarknet Backbone ---
    # Stem
    x = Cbs_Block(filters=8, kernel_size=3, strides=2, name='cbs__block')(x)
    
    # Stage 1
    x = Cbs_Block(filters=16, kernel_size=3, strides=2, name='cbs__block_1')(x)
    x = C2f_Block(filters=16, num_bottleneck=1, name='c2f__block')(x)
    
    # Stage 2
    x = Cbs_Block(filters=32, kernel_size=3, strides=2, name='cbs__block_2')(x)
    p3 = C2f_Block(filters=32, num_bottleneck=2, name='c2f__block_1')(x)
    
    # Stage 3
    x = Cbs_Block(filters=64, kernel_size=3, strides=2, name='cbs__block_3')(p3)
    p4 = C2f_Block(filters=64, num_bottleneck=2, name='c2f__block_2')(x)

    # Stage 4
    x = Cbs_Block(filters=128, kernel_size=3, strides=2, name='cbs__block_4')(p4)
    x = C2f_Block(filters=128, num_bottleneck=1, name='c2f__block_3')(x)
    p5 = Sppf_Block(filters=128, name='sppf__block')(x)

    # --- PANet Neck ---
    # FPN Top-Down
    p5_up_p4 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d')(p5)
    p4_merge = tf.keras.layers.Concatenate(axis=-1, name='concatenate')([p5_up_p4, p4])
    p4_fused = C2f_Block(filters=64, num_bottleneck=3, shortcut=False, name='c2f__block_4')(p4_merge)

    p4_up_p3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='up_sampling2d_1')(p4_fused)
    p3_merge = tf.keras.layers.Concatenate(axis=-1, name='concatenate_1')([p4_up_p3, p3])
    p3_fused_ph_1 = C2f_Block(filters=32, num_bottleneck=3, shortcut=False, name='c2f__block_5')(p3_merge)

    # PANet Bottom-Up
    p3_down_p4 = Cbs_Block(filters=32, kernel_size=3, strides=2, name='cbs__block_5')(p3_fused_ph_1)
    p4_merge2 = tf.keras.layers.Concatenate(axis=-1, name='concatenate_2')([p3_down_p4, p4_fused])
    p4_fused_ph_2 = C2f_Block(filters=64, num_bottleneck=3, shortcut=False, name='c2f__block_6')(p4_merge2)

    p4_down_p5 = Cbs_Block(filters=64, kernel_size=3, strides=2, name='cbs__block_6')(p4_fused_ph_2)
    p5_merge2 = tf.keras.layers.Concatenate(axis=-1, name='concatenate_3')([p4_down_p5, p5])
    p5_fused_ph_3 = C2f_Block(filters=128, num_bottleneck=3, shortcut=False, name='c2f__block_7')(p5_merge2)

    # --- Classification Head ---
    g3 = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(p3_fused_ph_1)
    g4 = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_1')(p4_fused_ph_2)

    flattened_x = tf.keras.layers.Reshape((49, 128), name='reshape')(p5_fused_ph_3)
    transformer_input_x = Static_Positional_Embedding(seq_len=49, d_model=128, name='static__positional__embedding')(flattened_x)

    # Reconstruct regularizer from hyperparameters dictionary
    reg_type = hps_dict.get('reg_type', 'None')
    if reg_type == 'l1':
        reg = tf.keras.regularizers.L1(l1=hps_dict.get('l1', 1e-5))
    elif reg_type == 'l2':
        reg = tf.keras.regularizers.L2(l2=hps_dict.get('l2', 1e-5))
    elif reg_type == 'l1_l2':
        reg = tf.keras.regularizers.L1L2(l1=hps_dict.get('l1', 1e-5), l2=hps_dict.get('l2', 1e-5))
    else:
        reg = None

    # Transformer Encoder Block
    transformer_output = Transformer_Encoder(
        num_heads=4, d_model=128, ff_dim=512, dropout=0.1, kernel_regularizer=reg, name='transformer__encoder'
    )(transformer_input_x)

    g5 = tf.keras.layers.GlobalAveragePooling1D(name='global_average_pooling1d')(transformer_output)
    merged_features = tf.keras.layers.Concatenate(axis=-1, name='concatenate_4')([g3, g4, g5])

    x_head = tf.keras.layers.Dropout(0.3, name='dropout')(merged_features)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg, name='dense')(x_head)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='yolov8_transformer_classifier')

    # Compile using learning rate from dictionary
    learning_rate = hps_dict.get('lr', 1e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model
