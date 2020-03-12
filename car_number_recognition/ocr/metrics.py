from argus.metrics import Metric

def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]


class StringAccuracy(Metric):
    name = "str_accuracy"
    better = "max"

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        a = len(preds)
        b = len(targets)
        print(a, b)
        for i in range(len(preds)):
            for j in range(min(len(preds[i]), len(targets[i]))):
                if preds[i][j] == targets[i][j]:
                    self.correct += 1
        # self.correct += 1
        self.count += 1
        # TODO: Count correct answers

    def compute(self):
        if self.count == 0:
            # raise Exception('Must be at least one example for computation')
            return 0
        return self.correct / self.count

# TODO: In the same way you can write Accuracy by position of letter
# or quality of negative examples and target


class CER(Metric):
    name = "char_error_rate"
    better = "min"

    def reset(self):
        # self.cer_s, self.cer_i, self.cer_d, self.cer_n = 0, 0, 0, 0

        self.cer_sum = 0
        self.count = 0

    def update(self, step_output: dict):
        hyp = step_output["prediction"]
        ref = step_output["target"]
        cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
        for n in range(len(ref)):
            # update CER statistics
            _, (s, i, d) = levenshtein(ref[n], hyp[n])
            cer_s += s
            cer_i += i
            cer_d += d
            cer_n += len(ref[n])
        self.cer_sum += 100.0 * (cer_s + cer_i + cer_d) / cer_n
        self.count += 1
        # TODO: Count correct answers

    def compute(self):
        if self.count == 0:
            # raise Exception('Must be at least one example for computation')
            return 0
        return self.cer_sum / self.count