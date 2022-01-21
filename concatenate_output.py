from data.dataset import OutputDataset

output_path = './results/sddm_wsj0_spec_220119_140239/results/'

datatype = '.spec.npy'
sample_rate = 16000

output_dataset = OutputDataset(output_path, datatype, sample_rate)

for i in range(len(output_dataset)):
    print(output_dataset.getName(i))

