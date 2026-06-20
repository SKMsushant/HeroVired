
# ==========================================
# CELL 0
# ==========================================
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import keras_tuner as kt

tf.keras.utils.set_random_seed(42)

start_time=time.time()

print(f'{50*'='} Dataset Folder Validation {50*'='}')
data_dir=r'D:\D\adsw\DSA\hvc\capstone\Plant_leave_diseases_dataset_with_augmentation'

if  os.path.exists(data_dir):
    folders=[f for f in os.listdir(data_dir)]
    print(f'path exists! Found {len(folders)} class subfolders')
    print("Sample folder names:")
    for f in folders:
        print(f"  - {f}")
else:
    print(f"Error: Path '{data_dir}' does not exist. Please check the spelling.")

print(f'{50*'='} importing Dataset and creation {50*'='}')

full_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    batch_size=None,
    image_size=(224,224),
    shuffle=True,
    seed=42
)
class_names = full_ds.class_names
num_classes = len(class_names)
print("\n--- Dataset Info ---")
print(f"Total Images: {len(full_ds)}")
print(f"Total Classes: {num_classes}")


print(f'{50*'='} random dataset  validation {50*'='}')

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(full_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(class_names[label.numpy()], fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()


print(f'{50*'='} Train Test and Validation Data Creation {50*'='}')

dataset_size=len(full_ds)
train_size=int(0.8*dataset_size)
val_size=int(0.1*dataset_size)
test_size=dataset_size-train_size-val_size

print(f'Splits : Train : {train_size} Test : {test_size} Validation : {val_size}')

ds_train=full_ds.take(train_size)
remaining=full_ds.skip(train_size)
ds_val=remaining.take(val_size)
ds_test=remaining.skip(val_size)

print(f'{50*'='} Preprocessing , Train Test and validation Pipeline {50*'='}')

def preprocess_image(image,labels):
    image=tf.cast(image,tf.float32)/255.0
    return image,labels

batch_size=32
autotune=tf.data.AUTOTUNE

train_pipeline=ds_train.map(preprocess_image,num_parallel_calls=autotune)
train_pipeline=train_pipeline.cache()
train_pipeline=train_pipeline.shuffle(buffer_size=1000)
train_pipeline=train_pipeline.batch(batch_size)
train_pipeline=train_pipeline.prefetch(buffer_size=autotune)

val_pipeline=ds_val.map(preprocess_image,num_parallel_calls=autotune)
val_pipeline=val_pipeline.batch(batch_size)
val_pipeline=val_pipeline.cache()
val_pipeline=val_pipeline.prefetch(buffer_size=autotune)


test_pipeline=ds_test.map(preprocess_image,num_parallel_calls=autotune)
test_pipeline=test_pipeline.batch(batch_size)
test_pipeline=test_pipeline.cache()
test_pipeline=test_pipeline.prefetch(buffer_size=autotune)

end_time=time.time()
print(f'Total Time Taken : {(end_time-start_time)/60} Minutes')

print("Data pipeline optimization complete!")

# ==========================================
# CELL 2
# ==========================================
#### =============================================CBS Block=================================================
class Cbs_Block(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides=1,padding='same',**kwargs):

        super().__init__(**kwargs)
        self.conv=tf.keras.layers.Conv2D(filters,kernel_size,strides=strides,padding=padding,use_bias=False)
        self.bn=tf.keras.layers.BatchNormalization()
        self.act=tf.keras.layers.Activation('silu')

    def call(self,x,training=False):
        return self.act(self.bn(self.conv(x),training=training))


####===============================================C2f Block================================================
class C2f_Block(tf.keras.layers.Layer):
    def __init__(self,filters,num_bottleneck=1,shortcut=True,**kwargs):

        super().__init__(**kwargs)
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

####============================================Bottleneck Inside C2f Block=====================================

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self,filters,shortcut=True,**kwargs):
        super().__init__(**kwargs)
        self.cbs1=Cbs_Block(filters,kernel_size=3)
        self.cbs2=Cbs_Block(filters,kernel_size=3)
        self.shortcut=shortcut

    def call(self,x,training=False):
        out=self.cbs2(self.cbs1(x,training=training),training=training)
        return x+out if self.shortcut else out
    

####=============================================SPPF layer=======================================================
    
class Sppf_Block(tf.keras.layers.Layer):
    def __init__(self,filters,pool_size=5,**kwargs):
        super().__init__(**kwargs)
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

### ============================================Positional Enbedding ============================================== 

class Static_Positional_Embedding(tf.keras.layers.Layer):
    def __init__(self,seq_len,d_model,**kwargs):
        super().__init__(**kwargs)
        self.seq_len=seq_len
        self.d_model=d_model


        sentence_positional_vector_list=[]

        for word_position in range(self.seq_len):

            word_position_vector=[]
            for i in range(int((self.d_model/2))):
                y_sin=np.sin(word_position/10000**((2*i)/self.d_model))
                y_cos=np.cos(word_position/10000**((2*i)/self.d_model))
                word_position_vector.append(y_sin)
                word_position_vector.append(y_cos)

            sentence_positional_vector_list.append(word_position_vector)

        encoding_matrix= np.array(sentence_positional_vector_list)
        self.positional_encoding=tf.constant(encoding_matrix,dtype=tf.float32)[None,...]

    def call(self,x):
        return x+self.positional_encoding
    
####=============================================Transformer Encoder============================================================

class Transformer_Encoder(tf.keras.layers.Layer):

    def __init__(self,num_heads,d_model,ff_dim,dropout=0.1,kernel_regularizer=None,**kwargs):
        super().__init__(**kwargs)

        self.mha=tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=d_model,value_dim=d_model)
        self.ffn=tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim,activation='relu',kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dense(d_model)
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



# ==========================================
# CELL 4
# ==========================================
class Build_HyperCNNmodel(kt.HyperModel):

    def __init__(self,num_classes,model_save_path,img_size=224,**kwargs):
        super().__init__(**kwargs)
        self.num_classes=num_classes
        self.img_size=img_size
        self.model_save_path=model_save_path

    def build(self,hp):
        inputs=tf.keras.layers.Input(shape=(self.img_size,self.img_size,3))

    ## Yolo v8
        ## Data Augmentation
        x=tf.keras.layers.RandomFlip('horizontal_and_vertical')(inputs)
        x=tf.keras.layers.RandomRotation(0.2)(x)
        x=tf.keras.layers.RandomContrast(0.15)(x)
        x=tf.keras.layers.RandomZoom(0.1)(x)


        ### CSPDarknet
        #stem 112x112
        #0
        x=Cbs_Block(filters=8,kernel_size=3,strides=2)(x)
        
        ## stage1:56x56
        #1
        x=Cbs_Block(filters=16,kernel_size=3,strides=2)(x)
        #2
        x=C2f_Block(filters=16,num_bottleneck=1)(x)
        
        #stage2:28x28
        #3 
        x=Cbs_Block(filters=32,kernel_size=3,strides=2)(x)
        #4
        p3=C2f_Block(filters=32,num_bottleneck=2)(x)
        
        #stage3:14x14
        #5 
        x=Cbs_Block(filters=64,kernel_size=3,strides=2)(p3)
        #6
        p4=C2f_Block(filters=64,num_bottleneck=2)(x)

        #stage4:7x7
        #7
        x=Cbs_Block(filters=128,kernel_size=3,strides=2)(p4)
        #8
        x=C2f_Block(filters=128,num_bottleneck=1)(x)
        #9
        p5=Sppf_Block(filters=128)(x)



        # Neck(FPN+PAnet)

        #FPN Top-Down Path
        # Upsampling p3 to match p4 and then to p5 to match p5 to perform concatenation after each upsampling
        #p5-to-p4 (14x14)
        #10 
        p5_up_p4=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(p5)
        #11
        p4_merge=tf.keras.layers.Concatenate(axis=-1)([p5_up_p4,p4])
        #12
        p4_fused=C2f_Block(filters=64,num_bottleneck=3,shortcut=False)(p4_merge)

        #p4-to-p3 (28x28)
        #13
        p4_up_p3=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(p4_fused)
        #14
        p3_merge=tf.keras.layers.Concatenate(axis=-1)([p4_up_p3,p3])
        #15
        p3_fused_ph_1=C2f_Block(filters=32,num_bottleneck=3,shortcut=False)(p3_merge)

        ##PANet Bottom-Up Pathway
        ##Downsampling match p3 to p4 and p4 to p5
        #p3-to-p4 (14x14)
        #16
        p3_down_p4=Cbs_Block(filters=32,kernel_size=3,strides=2)(p3_fused_ph_1)
        #17
        p4_merge2=tf.keras.layers.Concatenate(axis=-1)([p3_down_p4,p4_fused])
        #18
        p4_fused_ph_2=C2f_Block(filters=64,num_bottleneck=3,shortcut=False)(p4_merge2)

        #p4-to-p5(7x7)
        #19
        p4_down_p5=Cbs_Block(filters=64,kernel_size=3,strides=2)(p4_fused_ph_2)
        #20
        p5_merge2=tf.keras.layers.Concatenate(axis=-1)([p4_down_p5,p5])
        #21
        p5_fused_ph_3=C2f_Block(filters=128,num_bottleneck=3,shortcut=False)(p5_merge2)


        ### Multi-Scale Classification Head & Transformers

        g3=tf.keras.layers.GlobalAveragePooling2D()(p3_fused_ph_1)
        g4=tf.keras.layers.GlobalAveragePooling2D()(p4_fused_ph_2)

        flattened_x=tf.keras.layers.Reshape((49,128))(p5_fused_ph_3)
        transformer_input_x=Static_Positional_Embedding(seq_len=49,d_model=128)(flattened_x)

        #==================
        hp=kt.HyperParameters()
        reg_type=hp.Choice('reg_type',['l1','l2','l1_l2','None'])
        if reg_type=='l1':
            reg=tf.keras.regularizers.L1(
                l1=hp.Float('l1',min_value=1e-1,max_value=5e-1,sampling='log')
            )
        elif reg_type=='l2':
            reg=tf.keras.regularizers.L2(
                l2=hp.Float('l2',min_value=1e-1,max_value=5e-1,sampling='log')
            )
        elif reg_type=='l1_l2':
            reg=tf.keras.regularizers.L1L2(
                l1=hp.Float('l1',min_value=1e-1,max_value=5e-1,sampling='log'),
                l2=hp.Float('l2',min_value=1e-1,max_value=5e-1,sampling='log')
            )
        else:
            reg=None
        #===============================

        transformer_output=Transformer_Encoder(num_heads=4,d_model=128,ff_dim=512,dropout=0.1,kernel_regularizer=reg)(transformer_input_x)

        g5=tf.keras.layers.GlobalAveragePooling1D()(transformer_output)
        merged_features=tf.keras.layers.Concatenate(axis=-1)([g3,g4,g5])

        x_head=tf.keras.layers.Dropout(0.3)(merged_features)
        outputs=tf.keras.layers.Dense(self.num_classes,activation='softmax')(x_head)

        model=tf.keras.Model(inputs=inputs,outputs=outputs,name='yolov8_transformer_classifier')

        learning_rate=hp.Float(name='lr',min_value=1e-4,max_value=1e-1,sampling='log')   
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model
    
    def fit(self,hp,model,x,y=None,*args,**kwargs):
        
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path,exist_ok=True)

        model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.model_save_path,"best_hypermodel.keras"),
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            save_freq='epoch'
        )

        early_stopping=tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.005,
            patience=5,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        callbacks=kwargs.pop('callbacks',[])
        callbacks.append(model_checkpoint)
        callbacks.append(early_stopping)
        callbacks.append(reduce_lr)

        return model.fit(
            x=x,
            y=None,
            epochs=hp.Int('epochs',min_value=10,max_value=30,step=5),
            callbacks=callbacks,
            verbose='auto',
            **kwargs
        )

# ==========================================
# CELL 6
# ==========================================
tuner_start=time.time()
hyperCNNmodel=Build_HyperCNNmodel(num_classes=num_classes,model_save_path=r'D:\D\adsw\DSA\hvc\capstone\models',img_size=224)


hp=kt.HyperParameters()
model_test=hyperCNNmodel.build(hp)
model_test.summary()


tuner_log_path=r"D:\D\adsw\DSA\hvc\capstone\logs\tuner_logs"
if not os.path.exists(tuner_log_path):
    os.makedirs(tuner_log_path,exist_ok=True)

param_tuner=kt.BayesianOptimization(
    hypermodel=hyperCNNmodel,
    objective='val_accuracy',
    max_trials=5,
    seed=42,
    executions_per_trial=1,
    overwrite=True,
    directory=tuner_log_path,
    project_name='yolv8_transformer_tuner'
)

param_tuner.search(
    x=train_pipeline,
    validation_data=val_pipeline
)

tuner_end=time.time()
print(f'{(tuner_end-tuner_start)/60} Minutes')

# ==========================================
# CELL 7
# ==========================================

