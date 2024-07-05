import logging
import torch
import re
import os

from transformers import AdamW

class Model(torch.nn.Module):
    def __init__(self, args, inputs):
        super().__init__()
        self.args = args
        self.optimizer = None

    @property
    def device(self):
        if self.args.train.gpu:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        all_params = set(self.parameters())
        params = [{"params": list(all_params), "lr": self.args.lr.bert}]
        self.optimizer = AdamW(params)


    def update_best(self, best, summary, epoch):
        stop_key = 'eval_test_{}'.format(self.args.train.stop)
        # stop_key = 'eval_dev_{}'.format(self.args.train.stop)
        # train_key = 'eval_train_{}'.format(self.args.train.stop) #没有在train数据集上eval, 所以此项为空
        if self.args.train.not_eval or (best.get(stop_key, 0) <= summary[stop_key] and self.args.train.stopmin is None) or (best.get(stop_key, summary[stop_key]) >= summary[stop_key] and self.args.train.stopmin is not None):
            if self.args.train.not_eval:
                best.update(summary)
                if self.args.train.max_save > 0:
                    self.save_model('epoch={epoch}'.format(epoch = epoch))
                    self.clear_saves()
            else:
                best_test = '{:f}'.format(summary[stop_key]) # :f浮点数格式化
                # best_dev = '{:f}'.format(summary[stop_key])
                # best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                if self.args.train.max_save > 0:
                    # self.save_model('epoch={epoch},train_{key}={train},dev_{key}={dev}'.format(epoch = epoch, train = best_train, dev = best_dev, key = self.args.train.stop))
                    self.save_model('epoch={epoch},test_{key}={test}'.format(epoch = epoch, test = best_test, key = self.args.train.stop))
                    self.clear_saves()
        return best


    def save_model(self, name):
        file = "{}/{}.pkl".format(self.args.dir.output, name)
        if not os.path.exists(self.args.dir.output):
            os.makedirs(self.args.dir.output)
        logging.info("Saving models to {}".format(name))
        state = {
            "models": self.state_dict()
        }
        torch.save(state, file)
        #TODO 修改
        file1 = "saved/best_model.pkl"
        state1 = {
            "models": self.state_dict()
        }
        torch.save(state1, file1)


    def clear_saves(self):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > self.args.train.max_save:
            for score, name in scores_and_files[self.args.train.max_save : ]:
                os.remove(name)


    def get_saves(self):
        files = [f for f in os.listdir(self.args.dir.output) if f.endswith('.pkl')]
        scores = []
        for name in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.train.stop)
            dev_acc = re.findall(re_str, name)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(self.args.dir.output, name)))
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores


    def load(self, file):
        logging.info("Loading models from {}".format(file))
        state = torch.load(file)
        model_state = state["models"]
        self.load_state_dict(model_state)


    def get_summary(self, epoch, iteration):
        return {"epoch": epoch, "iteration": iteration}

    # @staticmethod
    # def record(pred, dataset, set_name, args):
    #     pass


    def pred_flatten(self, out):
        pred = []
        for ele in out:
            pred.append(ele)
        return pred


    def get_max_train_batch(self, dataset):
        if self.args.dataset.part: # 为0时也不限制总的batch数
            max_train = min(self.args.dataset.part, len(dataset))
        else:
            max_train = len(dataset) # slice语法的end可以无限大,会自动截断
        return max_train