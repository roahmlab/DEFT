import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""example of reading pkl files"""

clamp_type = "middle"
training_case = 1
BDLO_type = 3
eval_loss_1 = np.array(pd.read_pickle(r"../training_record/eval_%s_loss_DEFT_%s_%s.pkl" % (clamp_type, training_case, BDLO_type)))
eval_step_1 = np.array(pd.read_pickle(r"../training_record/eval_%s_epoches_DEFT_%s_%s.pkl" % (clamp_type, training_case, BDLO_type)))
print(np.sqrt(eval_loss_1))
print("loss minimum: ", np.min(np.sqrt(eval_loss_1)))
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(10)
fig.set_figwidth(20)


line1 = ax2.plot(eval_step_1, np.sqrt(eval_loss_1), label='%s'%BDLO_type)

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


