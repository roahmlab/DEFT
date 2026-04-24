import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""example of reading pkl files"""

clamp_type = "ends"
# model = "TreeLSTM"
# model = "GCN"
model = "DEFT"
training_case = 1
BDLO_type = 6
eval_loss_1 = np.array(pd.read_pickle(r"../training_record/eval_%s_loss_%s_%s_%s.pkl" % (clamp_type, model, training_case, BDLO_type)))
eval_step_1 = np.array(pd.read_pickle(r"../training_record/eval_%s_epoches_%s_%s_%s.pkl" % (clamp_type,  model, training_case, BDLO_type)))
eval_loss_plot = eval_loss_1 if BDLO_type == 6 else np.sqrt(eval_loss_1)
print(eval_loss_plot)
print("loss minimum: ", np.min(eval_loss_plot))
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(10)
fig.set_figwidth(20)

line1 = ax2.plot(eval_step_1, eval_loss_plot, label='%s'%BDLO_type)

# # # #
ax1.set_title('BDLO1: Training')
ax1.set_xlabel('Training Iterations')
ax1.set_ylabel('MSE')

ax2.set_title('BDLO1: Eval')
ax2.set_xlabel('Training Iterations')
ax1.set_ylabel('MSE')

ax1.grid(which = "minor")
ax1.minorticks_on()
ax2.grid(which = "minor")
ax2.minorticks_on()
plt.legend()
plt.show()


