import json
import os
import random


def sample_MATH(path):
    types = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
             "prealgebra", "precalculus"]
    mathData = {}
    levelCount = [0,0,0,0,0]
    sampleConstrain = 20
    index = 0
    while index < len(types):
        if sum(levelCount) == sampleConstrain * len(levelCount):
            break
        allFiles = os.listdir(path + "/" + types[index])
        random.shuffle(allFiles)
        for file in allFiles:
            filePath = path + "/" + types[index] + "/" + file
            problem = json.load(open(filePath))
            level = int(problem['level'].split(" ")[1])-1
            if levelCount[level] < sampleConstrain:
                if level not in mathData.keys():
                    mathData[level] = {}
                    mathData[level][types[index]] = [problem]
                elif types[index] not in mathData[level].keys():
                    mathData[level][types[index]] = [problem]
                else:
                    mathData[level][types[index]].append(problem)
                levelCount[level] += 1
            index += 1
            break
        index = index % len(types)
    json.dump(mathData,open("test_set/math_subset_20.json","w"))

def sample_chess(path,size=50):
    chess_data = json.load(open(path,"r"))
    samples = random.sample(chess_data['examples'],k=size)
    for sample in samples:
        sample['move'] = sample['input'].split(" ")[-1]
    return samples

def loadMath(path):
    return json.load(open(path))


if __name__ == "__main__":
    sample_chess("chess_state_tracking/synthetic_short/task.json")
