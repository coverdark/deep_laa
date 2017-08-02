# deep_laa
## Files and datasets
Run laa-s.py (according to LAA-B) or laa_z_y.py (according to LAA-O and LAA-L).

Three datasets are available: Bluebirds (bluebird_data.npz), Flowers (flower_data.npz), and Web Search (web_processed_data_feature_2.npz).
You can uncomment the filename to select the corresponding dataset.

## Key model parameters are:
laa-s.py:
```
loss_classifier = loss_classifier_y_x \
     + 0.0001 * loss_y_kl \
     + 0.005/source_num/category_size/category_size * loss_w_classifier_l1
...
learning_rate = 0.005
```

laa_z_y.py:
```
n_z = 1 # number of latent aspects
...
loss_classifier = loss_classifier_y_x \
     + 0.0001*loss_y_kl \
     + 0.005/source_num/category_size/category_size * (loss_w_classifier_l1 + loss_b_classifier_l1 + loss_w_decoder_l1 + loss_b_decoder_l1) \
     + 0.5/source_num/n_z/n_z * (loss_z_weights_l2 + loss_z_biases_l2)
...
learning_rate = 0.01
```
## Recommended parameters (for laa_z_y.py)
Bluebirds:
```
n_z = 2
...
loss_classifier = loss_classifier_y_x \
     + 0.0001*loss_y_kl \
     + 0.005/source_num/category_size/category_size * (loss_w_classifier_l1 + loss_b_classifier_l1 + loss_w_decoder_l1 + loss_b_decoder_l1) \
     + 0.5/source_num/n_z/n_z * (loss_z_weights_l2 + loss_z_biases_l2)
...
learning_rate = 0.01
```

Flowers:
```
n_z = 2
...
loss_classifier = loss_classifier_y_x \
     + 0.0001*loss_y_kl \
     + 0.05/source_num/category_size/category_size * (loss_w_classifier_l1 + loss_b_classifier_l1 + loss_w_decoder_l1 + loss_b_decoder_l1) \
     + 0.1/source_num/n_z/n_z * (loss_z_weights_l2 + loss_z_biases_l2)
...
learning_rate = 0.001
```

Web Search:
```
n_z = 1
...
loss_classifier = loss_classifier_y_x \
     + 0.0001*loss_y_kl \
     + 0.005/source_num/category_size/category_size * (loss_w_classifier_l1 + loss_b_classifier_l1 + loss_w_decoder_l1 + loss_b_decoder_l1) \
     + 0.5/source_num/n_z/n_z * (loss_z_weights_l2 + loss_z_biases_l2)
...
learning_rate = 0.01
```