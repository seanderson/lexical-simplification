from nn import newsela_complex
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def analyze():
    LABELS = ["Cos>0.9", "SameTag", "Cos>0.8", "SameTag", "Cos>0.7", "SameTag", "Cos>0.6", "SameTag", "Cos>0.5",
              "SameTag", "Cos>0.4", "SameTag", "Cos>0.3", "SameTag", "Cos>0.27", "SameTag", "Cos>0.25", "SameTag", "Cos>0.23", "SameTag",
              "Cos>0.2", "SameTag"]
    # LABELS = ["Cos>0.4", "Cos>0.3", "Cos>0.25"]
    with open(newsela_complex.CHRIS_DATA) as file:
        scores = [int(x.split('\t')[2]) for x in file.readlines()]
    with open(newsela_complex.OUTPUT_FILE) as file:
        densities = [[float(y) for y in x.split('\t')[1:]]
                     for x in file.readlines()]
    if len(scores) != len(densities):
        print("Lengths are not equal...")
        exit(-1)
    best = ""
    best_value = 0

    i = 0
    while i < len(densities):
        if densities[i][0] == -1:
            del densities[i]
            del scores[i]
        else:
            i += 1

    for i in range(len(LABELS)):
        pred = [x[i] for x in densities]
        # plt.scatter(scores, pred)
        # plt.show()
        coeff, p_val = pearsonr(scores, pred)
        print(LABELS[i] + ": " + str(coeff) + " " + str(p_val))
        if abs(coeff) > best_value:
            best_value = abs(coeff)
            best = LABELS[i]
    print("Best metric is: " + best)
    average = [[[0, 0] for _ in range(len(LABELS))] for _ in range(10)]
    for i in range(len(densities)):
        if densities[i][0] == -1:
            continue
        for j in range(len(densities[i])):
            if scores[i] >= 10:
                pass
                continue
            average[scores[i]][j][0] += densities[i][j]
            average[scores[i]][j][1] += 1
    average = [[d[0]/d[1] for d in s] for s in average]
    print("Lvl\t" + "\t".join(LABELS))
    for i in range(len(average)):
        for j in range(len(average[i])):
            average[i][j] = str(round(average[i][j], 1)) + " " * (7 - len(str(round(average[i][j], 1))))
        print(str(i) + '\t' + '\t'.join(average[i]))


if __name__ == "__main__":
    analyze()