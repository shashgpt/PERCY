from config import *
from scripts.training.utils.teacher_network import *

class Fit(object):
    def __init__(self, config):
        self.config = config
    
    def calculate_distillation_strength(self, cur_iter): # Teacher Network
        # k,lb = params[0], params[1]
        k = 0.95
        lb = 0
        pi = 1. - max([k**cur_iter, lb])
        return pi
    
    def fit_no_distillation(self, train_dataset, validation_dataset, model, optimizer, loss, k_fold=None, l_fold=None, additional_validation_datasets=None):
    
        # Collects per-epoch loss and acc like Keras' fit().
        history = {} 
        history['loss'] = []
        history['val_loss'] = []
        history['accuracy'] = []
        history['val_accuracy'] = []
        if additional_validation_datasets!=None:
            for key, dataset_obj in additional_validation_datasets["dataset"].items():
                history[key+'_loss'] = []
                history[key+'_accuracy'] = []

        start_time_sec = time.time()

        patience = 0
        if self.config["metric"] == "val_accuracy":
            best_val_acc = 0
        elif self.config["metric"] == "val_loss":
            best_val_loss = 0
        
        # Start training
        for epoch in range(1, self.config["train_epochs"]+1):

            # Train iteration
            print("\n")
            train_loader = DataLoader(train_dataset, batch_size = self.config["mini_batch_size"], shuffle=False)
            model.train()
            train_loss = 0.0
            num_train_correct  = 0
            num_train_examples = 0
            for batch_idx, input in enumerate(tqdm(train_loader)):
                input_data = input[0]
                target = input[1]
                input_data, target = input_data.to(self.config["device"]), target.to(self.config["device"])
                optimizer.zero_grad()
                softmax = nn.Softmax(dim=1)
                model_output = softmax(model(input_data))
                ground_truth = torch.nn.functional.one_hot(target)
                NLLLoss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))
                # log_softmax = nn.LogSoftmax(dim=1)
                # model_output = log_softmax(model(input_data))
                # NLLLoss = loss(model_output, target)
                NLLLoss.backward()
                optimizer.step()
                train_loss += NLLLoss.data.item() * input_data.size(0)
                num_train_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                num_train_examples += input_data.shape[0]
            train_acc = num_train_correct / num_train_examples
            train_loss = train_loss / len(train_loader.dataset)
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)

            # Validation iteration
            val_loader = DataLoader(validation_dataset, batch_size = self.config["mini_batch_size"], shuffle=False)
            model.eval()
            val_loss = 0.0
            num_val_correct  = 0
            num_val_examples = 0
            with torch.no_grad():
                for batch_idx, input in enumerate(tqdm(val_loader)):
                    input_data = input[0]
                    target = input[1]
                    input_data, target = input_data.to(self.config["device"]), target.to(self.config["device"])
                    softmax = nn.Softmax(dim=1)
                    model_output = softmax(model(input_data))
                    ground_truth = torch.nn.functional.one_hot(target)
                    NLLLoss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))
                    val_loss += NLLLoss.data.item() * input_data.size(0)
                    num_val_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                    num_val_examples += input_data.shape[0]
            val_acc  = num_val_correct / num_val_examples
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            # Print the log
            if epoch % 1 == 0:
                print("\nEpoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f" % \
                    (epoch, self.config["train_epochs"], train_loss, train_acc, val_loss, val_acc))

            # Additional Validation iterations
            if additional_validation_datasets!=None:
                for key, dataset_obj in additional_validation_datasets["dataset"].items():
                    loader = DataLoader(dataset_obj, batch_size = self.config["mini_batch_size"], shuffle=False)
                    model.eval()
                    loss = 0.0
                    num_correct  = 0
                    num_examples = 0
                    with torch.no_grad():
                        for batch_idx, input in enumerate(loader):
                            input_data = input[0]
                            target = input[1]
                            input_data, target = input_data.to(self.config["device"]), target.to(self.config["device"])
                            softmax = nn.Softmax(dim=1)
                            model_output = softmax(model(input_data))
                            ground_truth = torch.nn.functional.one_hot(target)
                            NLLLoss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))
                            loss += NLLLoss.data.item() * input_data.size(0)
                            num_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                            num_examples += input_data.shape[0]
                    acc  = num_correct / num_val_examples
                    loss = loss / len(loader.dataset)
                    history[key+'_loss'].append(loss)
                    history[key+'_accuracy'].append(acc)             
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
                    os.makedirs("assets/trained_models/"+self.config["asset_name"])
                if k_fold != None and l_fold != None:
                    torch.save(model.state_dict(), "assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".pt")
                else:
                    torch.save(model.state_dict(), "assets/trained_models/"+self.config["asset_name"]+".pt")
            elif val_acc <= best_val_acc:
                patience += 1
                if patience > self.config["patience"]:
                    if "early_stopping" in self.config["callbacks"]: 
                        break
                    else:
                        continue

        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.config["train_epochs"]
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

        return history
    
    def fit_distillation(self, train_dataset, val_dataset, model, optimizer, loss, train_dataset_rule_features, val_dataset_rule_features, k_fold=None, l_fold=None, additional_validation_datasets=None):

        # Collects per-epoch loss and acc like Keras' fit().
        history = {} 
        history['loss'] = []
        history['val_loss'] = []
        history['accuracy'] = []
        history['val_accuracy'] = []
        if additional_validation_datasets!=None:
            for key, dataset_obj in additional_validation_datasets["dataset"].items():
                history[key+'_loss'] = []
                history[key+'_accuracy'] = []

        start_time_sec = time.time()

        patience = 0
        if self.config["metric"] == "val_accuracy":
            best_val_acc = 0
        elif self.config["metric"] == "val_loss":
            best_val_loss = 0
        
        # Start training
        for epoch in range(1, self.config["train_epochs"]+1):

            # Train iteration
            print("\n")
            model.train()
            train_loss = 0.0
            num_train_correct  = 0
            num_train_examples = 0
            train_loader = DataLoader(train_dataset, batch_size = self.config["mini_batch_size"], shuffle=False)
            train_rule_features_loader = DataLoader(train_dataset_rule_features, batch_size = self.config["mini_batch_size"], shuffle=False)
            for batch_idx, (input, features) in enumerate(tqdm(zip(train_loader, train_rule_features_loader))):
                input_data = input[0]
                target = input[1]
                rule_features = features[0]
                rule_features_ind = features[1]
                input_data, target = input_data.to(self.config["device"]), target.to(self.config["device"])
                rule_features, rule_features_ind = rule_features.to(self.config["device"]), rule_features_ind.to(self.config["device"])
                rule_features_ind = rule_features_ind.reshape(rule_features_ind.shape[0], 1)
                optimizer.zero_grad()
                curr_epoch_num_train_batches = len(train_loader)
                curr_iteration = epoch*curr_epoch_num_train_batches+batch_idx
                distillation_strength = self.calculate_distillation_strength(curr_iteration*1./curr_epoch_num_train_batches)
                softmax = nn.Softmax(dim=1)
                model_output = softmax(model(input_data))
                f_but_y_pred_p = softmax(model(rule_features))
                f_but_full = torch.cat([rule_features_ind, f_but_y_pred_p], dim=1)
                rules = [FOL_A_but_B_pytorch(classes = self.config["classes"], input = input_data, features = f_but_full)]
                class_object = Teacher_network_pytorch(batch_size = self.config["mini_batch_size"], classes = self.config["classes"], rules = rules, device = self.config["device"])
                teacher_output = class_object.teacher_output(student_output = model_output)
                ground_truth = torch.nn.functional.one_hot(target)
                NLL_student_loss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))
                NLL_soft_loss = -torch.mean(torch.sum(teacher_output*torch.log(model_output), dim=1))
                distillation_loss = (1.0 - distillation_strength)*NLL_student_loss + distillation_strength*NLL_soft_loss
                distillation_loss.backward()
                optimizer.step()
                train_loss += distillation_loss.data.item() * input_data.size(0)
                num_train_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                num_train_examples += input_data.shape[0]
            train_acc = num_train_correct / num_train_examples
            train_loss = train_loss / len(train_loader.dataset)
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)

            # Validation iteration
            model.eval()
            val_loss = 0.0
            num_val_correct  = 0
            num_val_examples = 0
            val_loader = DataLoader(val_dataset, batch_size = self.config["mini_batch_size"], shuffle=False)
            val_rule_features_loader = DataLoader(val_dataset_rule_features, batch_size = self.config["mini_batch_size"], shuffle=False)
            with torch.no_grad():
                for batch_idx, (input, features) in enumerate(tqdm(zip(val_loader, val_rule_features_loader))):
                    input_data = input[0]
                    target = input[1]
                    rule_features = features[0]
                    rule_features_ind = features[1]
                    input_data, target = input_data.to(self.config["device"]), target.to(self.config["device"])
                    rule_features, rule_features_ind = rule_features.to(self.config["device"]), rule_features_ind.to(self.config["device"])
                    rule_features_ind = rule_features_ind.reshape(rule_features_ind.shape[0], 1)
                    curr_epoch_num_train_batches = len(val_loader)
                    curr_iteration = epoch*curr_epoch_num_train_batches+batch_idx
                    distillation_strength = self.calculate_distillation_strength(curr_iteration*1./curr_epoch_num_train_batches)
                    softmax = nn.Softmax(dim=1)
                    model_output = softmax(model(input_data))
                    f_but_y_pred_p = softmax(model(rule_features))
                    f_but_full = torch.cat([rule_features_ind, f_but_y_pred_p], dim=1)
                    rules = [FOL_A_but_B_pytorch(classes = self.config["classes"], input = input_data, features = f_but_full)]
                    class_object = Teacher_network_pytorch(batch_size = self.config["mini_batch_size"], classes = self.config["classes"], rules = rules, device = self.config["device"])
                    teacher_output = class_object.teacher_output(student_output = model_output)
                    ground_truth = torch.nn.functional.one_hot(target)
                    NLL_student_loss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))
                    NLL_soft_loss = -torch.mean(torch.sum(teacher_output*torch.log(model_output), dim=1))
                    distillation_loss = (1.0 - distillation_strength)*NLL_student_loss + distillation_strength*NLL_soft_loss
                    val_loss += distillation_loss.data.item() * input_data.size(0)
                    num_val_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                    num_val_examples += input_data.shape[0]
            val_acc  = num_val_correct / num_val_examples
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            if epoch % 1 == 0:
                print("\nEpoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f" % \
                        (epoch, self.config["train_epochs"], train_loss, train_acc, val_loss, val_acc))
            
            # Additional Validation iterations
            if additional_validation_datasets!=None:
                for key, dataset_obj in additional_validation_datasets["dataset"].items():
                    loader = DataLoader(dataset_obj, batch_size = self.config["mini_batch_size"], shuffle=False)
                    model.eval()
                    loss = 0.0
                    num_correct  = 0
                    num_examples = 0
                    with torch.no_grad():
                        for batch_idx, input in enumerate(loader):
                            input_data = input[0]
                            target = input[1]
                            input_data, target = input_data.to(self.config["device"]), target.to(self.config["device"])
                            softmax = nn.Softmax(dim=1)
                            model_output = softmax(model(input_data))
                            ground_truth = torch.nn.functional.one_hot(target)
                            NLLLoss = -torch.mean(torch.sum(ground_truth*torch.log(model_output), dim=1))
                            loss += NLLLoss.data.item() * input_data.size(0)
                            num_correct  += (torch.max(model_output, 1)[1] == target).sum().item()
                            num_examples += input_data.shape[0]
                    acc  = num_correct / num_val_examples
                    loss = loss / len(loader.dataset)
                    history[key+'_loss'].append(loss)
                    history[key+'_accuracy'].append(acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
                    os.makedirs("assets/trained_models/"+self.config["asset_name"])
                if k_fold != None and l_fold != None:
                    torch.save(model.state_dict(), "assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".pt")
                else:
                    torch.save(model.state_dict(), "assets/trained_models/"+self.config["asset_name"]+".pt")
            elif val_acc <= best_val_acc:
                patience += 1
                if patience > self.config["patience"]:
                    if "early_stopping" in self.config["callbacks"]: 
                        break
                    else:
                        continue

        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.config["train_epochs"]
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

        return history
        