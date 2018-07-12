from nn import newsela_complex


def analyze():
    with open(newsela_complex.CHRIS_DATA) as file:
        scores = [int(x.split('\t')[2]) for x in file.readlines()]
    with open(newsela_complex.OUTPUT_FILE) as file:
        densities = [[float(y) for y in x.split('\t')[1:]]
                     for x in file.readlines()]
    if len(scores) != len(densities):
        print("Lengths are not equal...")
        exit(-1)
    average = [[[0, 1] for x in densities[0]] for y in range(10)]
    for i in range(len(densities)):
        for j in range(len(densities[i])):
            if scores[i] >= 10:
                pass
                continue
            average[scores[i]][j][0] += densities[i][j]
            average[scores[i]][j][1] += 1
    average = [[d[0]/d[1] for d in s] for s in average]
    for x in average:
        print(x)


if __name__ == "__main__":
    analyze()