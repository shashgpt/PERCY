from config import *

class FOL_rules_pytorch(object):
    def __init__(self, classes, input, features):
        self.classes = classes
        self.input = input
        self.features = features

class FOL_A_but_B_pytorch(FOL_rules_pytorch):
    def __init__(self, classes, input, features):
        assert classes == 2
        super(FOL_A_but_B_pytorch, self).__init__(classes, input, features)

    def log_distribution(self, w, X=None, F=None):
        if F == None:
            X, F = self.input, self.features
        F_mask = F[:,0] # f_but_ind
        F_fea = F[:,1:] # f_but_y_pred_p
        distr_y0 = w*F_mask*F_fea[:,0] # y = 0 
        distr_y1 = w*F_mask*F_fea[:,1] # y = 1
        distr_y0 = distr_y0.reshape([distr_y0.shape[0],1])
        distr_y1 = distr_y1.reshape([distr_y1.shape[0],1])
        distr = torch.cat((distr_y0, distr_y1), dim=1)
        return distr

class Teacher_network_pytorch(object):
    def __init__(self, batch_size, classes, rules, device, rules_lambda = [1.0], teacher_regularizer = 6.0):
        self.batch_size = batch_size
        self.classes = classes
        self.rules = rules
        self.rules_lambda = rules_lambda
        self.teacher_regularizer = teacher_regularizer
        self.device = device

    def calc_rule_constraints(self, rules, rules_lambda, teacher_regularizer, batch_size, classes, new_data=None, new_rule_fea=None):
        if new_rule_fea==None:
            new_rule_fea = [None]*len(rules)
        distr_all = torch.zeros(batch_size, classes).to(self.device)
        for i,rule in enumerate(rules):
            distr = rule.log_distribution(teacher_regularizer*rules_lambda[i], new_data, new_rule_fea[i])
            distr_all += distr
        distr_all += distr
        distr_y0 = distr_all[:,0]
        distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
        distr_y0_copies = distr_y0.repeat(1, distr_all.shape[1])
        distr_all = distr_all - distr_y0_copies
        distr_all = torch.maximum(torch.minimum(distr_all, torch.Tensor([60.]).to(self.device)), torch.Tensor([-60.]).to(self.device)) # truncate to avoid over-/under-flow
        distr_all = torch.exp(distr_all)
        return distr_all

    def teacher_output(self, student_output):
        distr = self.calc_rule_constraints(rules = self.rules, rules_lambda = self.rules_lambda, teacher_regularizer = self.teacher_regularizer, batch_size = self.batch_size, classes = self.classes)
        q_y_given_x = 1.0*student_output*distr
        teacher_output = q_y_given_x / torch.sum(q_y_given_x, dim=1).reshape((self.batch_size, 1))
        return teacher_output