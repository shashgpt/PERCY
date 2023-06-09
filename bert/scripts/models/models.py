from config import *

class FOL_rules(object):
    def __init__(self, classes, input, features):
        self.classes = classes
        self.input = input
        self.features = features

class FOL_A_but_B(FOL_rules):
    def __init__(self, classes, input, features):
        assert classes == 1
        super(FOL_A_but_B, self).__init__(classes, input, features)

    def log_distribution(self, w, batch_size, X=None, F=None):
        if F == None:
            X, F = self.input, self.features
        F_mask = F[:,0] #f_but_ind
        F_fea = F[:,1] #f_but_y_pred_p
        distr_y1 = tf.math.multiply(w, tf.math.multiply(F_mask, F_fea)) #y = 1 
        distr_y0 = tf.math.multiply(w, tf.math.multiply(F_mask, tf.math.subtract(1.0, F_fea))) #y = 0
        distr_y0 = tf.reshape(distr_y0, [batch_size, self.classes])
        distr_y1 = tf.reshape(distr_y1, [batch_size, self.classes])
        distr = tf.concat([distr_y0, distr_y1], axis=1)
        return distr

class Teacher_network(object):
    def __init__(self, batch_size, classes, rules, rules_lambda, teacher_regularizer):
        self.batch_size = batch_size
        self.classes = classes
        self.rules = rules
        self.rules_lambda = rules_lambda
        self.teacher_regularizer = teacher_regularizer

    def calc_rule_constraints(self, rules, rules_lambda, teacher_regularizer, batch_size, classes, new_data=None, new_rule_fea=None):
        if new_rule_fea==None:
            new_rule_fea = [None]*len(rules)
        distr_all = tf.zeros([batch_size, classes], dtype=tf.dtypes.float32)
        for i, rule in enumerate(rules):
            distr = rule.log_distribution(teacher_regularizer*rules_lambda[i], batch_size, new_data, new_rule_fea[i])
            distr_all = tf.math.add(distr_all, distr)
        distr_all = tf.math.add(distr_all, distr)
        distr_y0 = distr_all[:,0]
        distr_y0 = tf.reshape(distr_y0, [batch_size, 1])
        distr_y0_copies = tf.concat([tf.identity(distr_y0), tf.identity(distr_y0)], axis=1)
        distr_all = tf.math.subtract(distr_all, distr_y0_copies)
        distr_all = tf.math.maximum(tf.math.minimum(distr_all, tf.constant([60.])), tf.constant([-60.])) # truncate to avoid over-/under-flow
        distr_all = tf.math.exp(distr_all)
        return distr_all

    def teacher_output(self, student_output):
        distr = self.calc_rule_constraints(rules = self.rules, 
                                            rules_lambda = self.rules_lambda, 
                                            teacher_regularizer = self.teacher_regularizer, 
                                            batch_size = self.batch_size, 
                                            classes = self.classes)
        q_y_given_x = tf.math.multiply(student_output, distr)
        teacher_output = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
        teacher_output = teacher_output[:,1]
        return teacher_output

class iteration_tracker(tf.keras.metrics.Metric):
    def __init__(self, name='iteration', **kwargs):
        super(iteration_tracker, self).__init__(name=name, **kwargs)
        self.iteration = self.add_weight(name='iteration', initializer='zeros')

    def update_state(self, curr_iter, sample_weight=None):
        self.iteration.assign_add(curr_iter)

    def result(self):
        return self.iteration

    def reset_states(self):
        self.iteration.assign(self.iteration)

class distillation_loss(tf.keras.metrics.Metric):
    def __init__(self, name='iteration', **kwargs):
        super(distillation_loss, self).__init__(name=name, **kwargs)
        self.distillation_loss = self.add_weight(name='distillation_loss', initializer='zeros')

    def update_state(self, distillation_loss, sample_weight=None):
        self.distillation_loss.assign(distillation_loss)

    def result(self):
        return self.distillation_loss

    def reset_states(self):
        self.distillation_loss.assign(0)

acc_tracker_per_epoch = tf.keras.metrics.BinaryAccuracy(name="accuracy")
iteration_tracker_metric = iteration_tracker()
distillation_loss_metric = distillation_loss()

# class IKD(Model):

#     def compute_mask(self, inputs, mask=None):
#         # Also split the mask into 2 if it presents.
#         if mask is None:
#             return None
#         return tf.split(mask, 2, axis=1)

#     def train_step(self, data): # an iteration
#         x,  y = data
#         sentences = x[0][0]
#         att_masks = x[0][1]
#         rule_features = x[1][0]
#         rule_features_att_masks = x[1][1]
#         sentiment_labels = y[0]
#         rule_features_ind = y[1]

#         with tf.GradientTape() as tape: # Forward propagation and loss calculation

#             # # IKD from my understanding
#             # y_pred = self(sentences, training=True)  #Forward pass
#             # f_but_y_pred_p = self(rule_features, training=True)
#             # distr = tf.math.multiply(f_but_y_pred_p, rule_features_ind, name=None) #check
#             # distr = tf.math.maximum(tf.math.minimum(distr, tf.constant([60.])), tf.constant([-60.]))
#             # multiply_but_exp = tf.math.exp(distr) #check
#             # q_y_given_x = tf.math.multiply(y_pred, multiply_but_exp, name=None) #check
#             # teacher = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
#             # loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#             # loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#             # m = tf.math.multiply(self.iteration_tracker_metric.result(), 1./1408)
#             # e = tf.math.pow(0.95, m)
#             # max = tf.math.maximum(e, 0.0)
#             # distillation_str = tf.math.subtract(1.0, max)
#             # s1 = tf.math.subtract(1.0, distillation_str)
#             # l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
#             # l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
#             # loss_value = tf.math.add(l1, l2)

#             # IKD from authors code
#             y_pred = self([sentences, att_masks], training=True)
#             f_but_y_pred_p = self([rule_features, rule_features_att_masks], training=True)
#             f_but_full = tf.concat([rule_features_ind[0], f_but_y_pred_p], axis=1)
#             rules = [FOL_A_but_B(classes = 1, input = input, features = f_but_full)]
#             class_object = Teacher_network(batch_size = 50, classes = 1, rules = rules, rules_lambda = [1.0], teacher_regularizer = 1.0)
#             teacher = class_object.teacher_output(student_output = y_pred)
#             loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#             loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#             m = tf.math.multiply(iteration_tracker_metric.result(), 1./1408)
#             e = tf.math.pow(0.95, m)
#             max = tf.math.maximum(e, 0.0)
#             distillation_str = tf.math.subtract(1.0, max)
#             s1 = tf.math.subtract(1.0, distillation_str)
#             l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
#             l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
#             loss_value = tf.math.add(l1, l2)
#             # loss_value = loss_fn_data(sentiment_labels, y_pred)

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss_value, trainable_vars)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         # Compute our own metrics
#         distillation_loss_metric.update_state(loss_value)
#         acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
#         iteration_tracker_metric.update_state(1.0)
#         return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}

#     @property
#     def metrics(self):
#         # We list our `Metric` objects here so that `reset_states()` can be
#         # called automatically at the start of each epoch
#         # or at the start of `evaluate()`.
#         # If you don't implement this property, you have to call
#         # `reset_states()` yourself at the time of your choosing.
#         return [distillation_loss_metric, acc_tracker_per_epoch, iteration_tracker_metric]
    
#     def test_step(self, data):

#         x,  y = data
#         sentences = x[0][0]
#         att_masks = x[0][1]
#         rule_features = x[1][0]
#         rule_features_att_masks = x[1][1]
#         sentiment_labels = y[0]
#         rule_features_ind = y[1]

#         # Compute predictions
#         y_pred = self([sentences, att_masks], training=True)
#         loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#         l1 = loss_fn_data(sentiment_labels, y_pred)

#         # Compute our own metrics
#         distillation_loss_metric.update_state(l1)
#         acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
#         return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}

# def cnn(config): 

#     input_sentence = layers.Input(shape=(None,), dtype=tf.int32)
#     attention_mask = layers.Input(shape=(None,), dtype=tf.int32)

#     for layer in config["bert_encoder"].layers:
#         layer.trainable = config["fine_tune_word_embeddings"]
#     word_embeddings = config["bert_encoder"](input_sentence, attention_mask=attention_mask)[0]
#     word_embeddings_reshaped = tf.keras.backend.expand_dims(word_embeddings, axis=1) # batch_size x 1 x sent_len x embedding_dim

#     conv_1 = layers.Conv2D(filters = config["n_filters"], 
#                             kernel_size = (config["filter_sizes"][0], config["embedding_dim"]),
#                             strides = 1,
#                             dilation_rate = 1,
#                             padding = "valid",
#                             data_format = "channels_first",
#                             name = "conv2D_1")(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
#     conv1_reshaped = tf.keras.backend.squeeze(conv_1, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
#     conv1_reshaped_relu = layers.ReLU()(conv1_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
#     max_pool_1 = layers.GlobalMaxPooling1D(data_format="channels_first",
#                                             name="maxpool1D_1")(conv1_reshaped_relu) # batch size x n_filters

#     conv_2 = layers.Conv2D(filters = config["n_filters"], 
#                             kernel_size = (config["filter_sizes"][1], config["embedding_dim"]),
#                             strides = 1,
#                             dilation_rate = 1,
#                             padding = "valid",
#                             data_format = "channels_first",
#                             name = "conv2D_2")(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
#     conv2_reshaped = tf.keras.backend.squeeze(conv_2, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
#     conv2_reshaped_relu = layers.ReLU()(conv2_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
#     max_pool_2 = layers.GlobalMaxPooling1D(data_format="channels_first",
#                                             name="maxpool1D_2")(conv2_reshaped_relu) # batch size x n_filters

#     conv_3 = layers.Conv2D(filters = config["n_filters"], 
#                             kernel_size = (config["filter_sizes"][2], config["embedding_dim"]),
#                             strides = 1,
#                             dilation_rate = 1,
#                             padding = "valid",
#                             data_format = "channels_first",
#                             name = "conv2D_3")(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
#     conv3_reshaped = tf.keras.backend.squeeze(conv_3, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
#     conv3_reshaped_relu = layers.ReLU()(conv3_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
#     max_pool_3 = layers.GlobalMaxPooling1D(data_format="channels_first",
#                                             name="maxpool1D_3")(conv3_reshaped_relu) # batch size x n_filters

#     concat = layers.Concatenate(axis=1, name="concatenate")([max_pool_1, max_pool_2, max_pool_3])
#     concat_dropout = layers.Dropout(rate=config["dropout"], seed=config["seed_value"], name="dropout")(concat)       
#     out = layers.Dense(1, activation='sigmoid', name='output')(concat_dropout)

#     if config["distillation"] == True:
#         model = IKD(inputs=[input_sentence, attention_mask], outputs=[out])
#     else:
#         model = Model(inputs=[input_sentence, attention_mask], outputs=[out])

#     if config["optimizer"] == "adam":
#         loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)    
#         model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss=loss, metrics=['accuracy'])
#     elif config["optimizer"] == "adadelta":
#         model.compile(tf.keras.optimizers.Adadelta(learning_rate=config["learning_rate"], rho=0.95, epsilon=1e-06), loss=loss, metrics=['accuracy'])
#     return model

class CNN(Model):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.bert_encoder = config["bert_encoder"]
        for layer in self.bert_encoder.layers:
            layer.trainable = config["fine_tune_word_embeddings"]

        self.conv_1 = layers.Conv2D(filters = config["n_filters"], 
                                    kernel_size = (config["filter_sizes"][0], config["embedding_dim"]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_1")
        self.conv1_reshaped_relu = layers.ReLU()
        self.max_pool_1 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                                    name="maxpool1D_1") # batch size x n_filters
        
        self.conv_2 = layers.Conv2D(filters = config["n_filters"], 
                                    kernel_size = (config["filter_sizes"][1], config["embedding_dim"]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_2")
        self.conv2_reshaped_relu = layers.ReLU()
        self.max_pool_2 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                                    name="maxpool1D_2")
        
        self.conv_3 = layers.Conv2D(filters = config["n_filters"], 
                                    kernel_size = (config["filter_sizes"][2], config["embedding_dim"]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_3")
        self.conv3_reshaped_relu = layers.ReLU()
        self.max_pool_3 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                            name="maxpool1D_3")
        
        self.concat = layers.Concatenate(axis=1, name="concatenate")
        self.concat_dropout = layers.Dropout(rate=config["dropout"], seed=config["seed_value"], name="dropout")   
        self.out = layers.Dense(1, activation='sigmoid', name='output')
    
    def compute_attention_masks(self, input_ids):
        zero = tf.constant(0, dtype=tf.int64)
        attention_masks = tf.cast(tf.not_equal(input_ids, zero), dtype=tf.int64)
        return attention_masks
    
    def call(self, input_ids, attention_masks=None, **kwargs):

        # Create attention masks
        if attention_masks == None:
            attention_masks = self.compute_attention_masks(input_ids)

        word_embeddings = self.bert_encoder(input_ids, attention_masks)[0]
        word_embeddings_reshaped = tf.keras.backend.expand_dims(word_embeddings, axis=1) # batch_size x 1 x sent_len x embedding_dim

        conv_1 = self.conv_1(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        conv1_reshaped = tf.keras.backend.squeeze(conv_1, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
        conv1_reshaped_relu = self.conv1_reshaped_relu(conv1_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
        max_pool_1 = self.max_pool_1(conv1_reshaped_relu) # batch size x n_filters

        conv_2 = self.conv_2(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        conv2_reshaped = tf.keras.backend.squeeze(conv_2, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
        conv2_reshaped_relu = self.conv2_reshaped_relu(conv2_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
        max_pool_2 = self.max_pool_2(conv2_reshaped_relu) # batch size x n_filters

        conv_3 = self.conv_3(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        conv3_reshaped = tf.keras.backend.squeeze(conv_3, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
        conv3_reshaped_relu = self.conv3_reshaped_relu(conv3_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
        max_pool_3 = self.max_pool_3(conv3_reshaped_relu) # batch size x n_filters

        concat = self.concat([max_pool_1, max_pool_2, max_pool_3])
        concat_dropout = self.concat_dropout(concat)       
        out = self.out(concat_dropout)

        return out



class LSTM(Model):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.bert_encoder = config["bert_encoder"]
        for layer in self.bert_encoder.layers:
            layer.trainable = config["fine_tune_word_embeddings"]

        # Classifier Layer
        self.lstm = layers.LSTM(config["hidden_units_seq_layer"], dropout=config["dropout"], name="classifier")
        self.out = layers.Dense(1, activation='sigmoid', name='output')
    
    def compute_attention_masks(self, input_ids):
        zero = tf.constant(0, dtype=tf.int64)
        attention_masks = tf.cast(tf.not_equal(input_ids, zero), dtype=tf.int64)
        return attention_masks
    
    def call(self, input_ids, attention_masks=None, **kwargs):

        # Create attention masks
        if attention_masks == None:
            attention_masks = self.compute_attention_masks(input_ids)
        word_embeddings = self.bert_encoder(input_ids, attention_masks)[0]
        out_1 = self.lstm(word_embeddings) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        out = self.out(out_1)
        return out



class CNN_IKD(Model):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.bert_encoder = config["bert_encoder"]
        for layer in self.bert_encoder.layers:
            layer.trainable = config["fine_tune_word_embeddings"]

        self.conv_1 = layers.Conv2D(filters = config["n_filters"], 
                                    kernel_size = (config["filter_sizes"][0], config["embedding_dim"]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_1")
        self.conv1_reshaped_relu = layers.ReLU()
        self.max_pool_1 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                                    name="maxpool1D_1") # batch size x n_filters
        
        self.conv_2 = layers.Conv2D(filters = config["n_filters"], 
                                    kernel_size = (config["filter_sizes"][1], config["embedding_dim"]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_2")
        self.conv2_reshaped_relu = layers.ReLU()
        self.max_pool_2 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                                    name="maxpool1D_2")
        
        self.conv_3 = layers.Conv2D(filters = config["n_filters"], 
                                    kernel_size = (config["filter_sizes"][2], config["embedding_dim"]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_3")
        self.conv3_reshaped_relu = layers.ReLU()
        self.max_pool_3 = layers.GlobalMaxPooling1D(data_format="channels_first",
                                            name="maxpool1D_3")
        
        self.concat = layers.Concatenate(axis=1, name="concatenate")
        self.concat_dropout = layers.Dropout(rate=config["dropout"], seed=config["seed_value"], name="dropout")   
        self.out = layers.Dense(1, activation='sigmoid', name='output')
    
    def compute_attention_masks(self, input_ids):
        zero = tf.constant(0, dtype=tf.int64)
        attention_masks = tf.cast(tf.not_equal(input_ids, zero), dtype=tf.int64)
        return attention_masks
    
    def call(self, input_ids, attention_masks=None, **kwargs):

        # Create attention masks
        if attention_masks == None:
            attention_masks = self.compute_attention_masks(input_ids)

        word_embeddings = self.bert_encoder(input_ids, attention_masks)[0]
        word_embeddings_reshaped = tf.keras.backend.expand_dims(word_embeddings, axis=1) # batch_size x 1 x sent_len x embedding_dim

        conv_1 = self.conv_1(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        conv1_reshaped = tf.keras.backend.squeeze(conv_1, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
        conv1_reshaped_relu = self.conv1_reshaped_relu(conv1_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
        max_pool_1 = self.max_pool_1(conv1_reshaped_relu) # batch size x n_filters

        conv_2 = self.conv_2(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        conv2_reshaped = tf.keras.backend.squeeze(conv_2, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
        conv2_reshaped_relu = self.conv2_reshaped_relu(conv2_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
        max_pool_2 = self.max_pool_2(conv2_reshaped_relu) # batch size x n_filters

        conv_3 = self.conv_3(word_embeddings_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        conv3_reshaped = tf.keras.backend.squeeze(conv_3, axis=3) # batch_size x 100 x sent len - filter_sizes[n] + 1
        conv3_reshaped_relu = self.conv3_reshaped_relu(conv3_reshaped) # batch_size x 100 x sent len - filter_sizes[n] + 1
        max_pool_3 = self.max_pool_3(conv3_reshaped_relu) # batch size x n_filters

        concat = self.concat([max_pool_1, max_pool_2, max_pool_3])
        concat_dropout = self.concat_dropout(concat)       
        out = self.out(concat_dropout)

        return out
    
    def train_step(self, data): # an iteration
        x,  y = data
        sentences = x[0][0]
        att_masks = x[0][1]
        rule_features = x[1][0]
        rule_features_att_masks = x[1][1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        with tf.GradientTape() as tape: # Forward propagation and loss calculation

            # # IKD from my understanding
            # y_pred = self(sentences, training=True)  #Forward pass
            # f_but_y_pred_p = self(rule_features, training=True)
            # distr = tf.math.multiply(f_but_y_pred_p, rule_features_ind, name=None) #check
            # distr = tf.math.maximum(tf.math.minimum(distr, tf.constant([60.])), tf.constant([-60.]))
            # multiply_but_exp = tf.math.exp(distr) #check
            # q_y_given_x = tf.math.multiply(y_pred, multiply_but_exp, name=None) #check
            # teacher = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
            # loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # m = tf.math.multiply(self.iteration_tracker_metric.result(), 1./1408)
            # e = tf.math.pow(0.95, m)
            # max = tf.math.maximum(e, 0.0)
            # distillation_str = tf.math.subtract(1.0, max)
            # s1 = tf.math.subtract(1.0, distillation_str)
            # l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            # l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            # loss_value = tf.math.add(l1, l2)

            # IKD from authors code
            y_pred = self([sentences, att_masks], training=True)
            f_but_y_pred_p = self([rule_features, rule_features_att_masks], training=True)
            f_but_full = tf.concat([rule_features_ind[0], f_but_y_pred_p], axis=1)
            rules = [FOL_A_but_B(classes = 1, input = input, features = f_but_full)]
            class_object = Teacher_network(batch_size = self.config["mini_batch_size"], classes = 1, rules = rules, rules_lambda = [1.0], teacher_regularizer = 1.0)
            teacher = class_object.teacher_output(student_output = y_pred)
            loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            m = tf.math.multiply(iteration_tracker_metric.result(), 1./1408)
            e = tf.math.pow(0.95, m)
            max = tf.math.maximum(e, 0.0)
            distillation_str = tf.math.subtract(1.0, max)
            s1 = tf.math.subtract(1.0, distillation_str)
            l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            loss_value = tf.math.add(l1, l2)
            # loss_value = loss_fn_data(sentiment_labels, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        distillation_loss_metric.update_state(loss_value)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        iteration_tracker_metric.update_state(1.0)
        return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [distillation_loss_metric, acc_tracker_per_epoch, iteration_tracker_metric]
    
    def test_step(self, data):

        x,  y = data
        sentences = x[0][0]
        att_masks = x[0][1]
        rule_features = x[1][0]
        rule_features_att_masks = x[1][1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        # Compute predictions
        y_pred = self([sentences, att_masks], training=True)
        loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        l1 = loss_fn_data(sentiment_labels, y_pred)

        # Compute our own metrics
        distillation_loss_metric.update_state(l1)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}

class LSTM_IKD(Model):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.bert_encoder = config["bert_encoder"]
        for layer in self.bert_encoder.layers:
            layer.trainable = config["fine_tune_word_embeddings"]

        # Classifier Layer
        self.lstm = layers.LSTM(config["hidden_units_seq_layer"], dropout=config["dropout"], name="classifier")
        self.out = layers.Dense(1, activation='sigmoid', name='output')
    
    def compute_attention_masks(self, input_ids):
        zero = tf.constant(0, dtype=tf.int64)
        attention_masks = tf.cast(tf.not_equal(input_ids, zero), dtype=tf.int64)
        return attention_masks
    
    def call(self, input_ids, attention_masks=None, **kwargs):

        # Create attention masks
        if attention_masks == None:
            attention_masks = self.compute_attention_masks(input_ids)
        word_embeddings = self.bert_encoder(input_ids, attention_masks)[0]
        out_1 = self.lstm(word_embeddings) # batch_size x 100 x sent len - filter_sizes[n] + 1 x 1
        out = self.out(out_1)
        return out
    
    def train_step(self, data): # an iteration
        x,  y = data
        sentences = x[0][0]
        att_masks = x[0][1]
        rule_features = x[1][0]
        rule_features_att_masks = x[1][1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        with tf.GradientTape() as tape: # Forward propagation and loss calculation

            # # IKD from my understanding
            # y_pred = self(sentences, training=True)  #Forward pass
            # f_but_y_pred_p = self(rule_features, training=True)
            # distr = tf.math.multiply(f_but_y_pred_p, rule_features_ind, name=None) #check
            # distr = tf.math.maximum(tf.math.minimum(distr, tf.constant([60.])), tf.constant([-60.]))
            # multiply_but_exp = tf.math.exp(distr) #check
            # q_y_given_x = tf.math.multiply(y_pred, multiply_but_exp, name=None) #check
            # teacher = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
            # loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # m = tf.math.multiply(self.iteration_tracker_metric.result(), 1./1408)
            # e = tf.math.pow(0.95, m)
            # max = tf.math.maximum(e, 0.0)
            # distillation_str = tf.math.subtract(1.0, max)
            # s1 = tf.math.subtract(1.0, distillation_str)
            # l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            # l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            # loss_value = tf.math.add(l1, l2)

            # IKD from authors code
            y_pred = self([sentences, att_masks], training=True)
            f_but_y_pred_p = self([rule_features, rule_features_att_masks], training=True)
            f_but_full = tf.concat([rule_features_ind[0], f_but_y_pred_p], axis=1)
            rules = [FOL_A_but_B(classes = 1, input = input, features = f_but_full)]
            class_object = Teacher_network(batch_size = self.config["mini_batch_size"], classes = 1, rules = rules, rules_lambda = [1.0], teacher_regularizer = 1.0)
            teacher = class_object.teacher_output(student_output = y_pred)
            loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            m = tf.math.multiply(iteration_tracker_metric.result(), 1./1408)
            e = tf.math.pow(0.95, m)
            max = tf.math.maximum(e, 0.0)
            distillation_str = tf.math.subtract(1.0, max)
            s1 = tf.math.subtract(1.0, distillation_str)
            l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            loss_value = tf.math.add(l1, l2)
            # loss_value = loss_fn_data(sentiment_labels, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        distillation_loss_metric.update_state(loss_value)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        iteration_tracker_metric.update_state(1.0)
        return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [distillation_loss_metric, acc_tracker_per_epoch, iteration_tracker_metric]
    
    def test_step(self, data):

        x,  y = data
        sentences = x[0][0]
        att_masks = x[0][1]
        rule_features = x[1][0]
        rule_features_att_masks = x[1][1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        # Compute predictions
        y_pred = self([sentences, att_masks], training=True)
        loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        l1 = loss_fn_data(sentiment_labels, y_pred)

        # Compute our own metrics
        distillation_loss_metric.update_state(l1)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}


def cnn(config):
    
    if config["distillation"] == True:
        model = CNN_IKD(config)
    elif config["distillation"] == False:
        model = CNN(config)

    if config["optimizer"] == "adam":
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss=loss, metrics=['accuracy'])
    elif config["optimizer"] == "adadelta":
        model.compile(tf.keras.optimizers.Adadelta(learning_rate=config["learning_rate"], rho=0.95, epsilon=1e-06), loss=loss, metrics=['accuracy'])
    return model

def lstm(config):
    
    if config["distillation"] == True:
        model = LSTM_IKD(config)
    elif config["distillation"] == False:
        model = LSTM(config)

    if config["optimizer"] == "adam":
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss=loss, metrics=['accuracy'])
    elif config["optimizer"] == "adadelta":
        model.compile(tf.keras.optimizers.Adadelta(learning_rate=config["learning_rate"], rho=0.95, epsilon=1e-06), loss=loss, metrics=['accuracy'])
    return model