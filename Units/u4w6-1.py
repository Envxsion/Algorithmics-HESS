'''The Subset Sum problem involves, for a given set of integers and target sum, determining whether there is a subset of the given set whose elements sum up to the target value. This can be solved with backtracking.
Write Python code based on the following pseudocode structure. It should return the current subset if the target is found, or null if the target cannot be found. Otherwise add the next element from the set to the current subset and call recursively. If the subset is not found, the element should be excluded and the call repeated with the next element.
function subsetSum(set, targetSum):
  return backtrack(set, targetSum, [], 0)
function backtrack(set, targetSum, currentSubset, currentIndex):
'''
def subsetSum(set, targetSum):
    return backtrack(set, targetSum, [], 0)

def backtrack(set, targetSum, currentSubset, currentIndex):
    if targetSum == 0:
        return currentSubset
    if targetSum < 0 or currentIndex >= len(set):
        return None
    
    print("Set: "+str(set) + " New sum: "+ str(targetSum -set[currentIndex]) + " Unknown: " + str(currentSubset + [set[currentIndex]]), " New idx: " + str(currentIndex+1) )
    withElement = backtrack(set, targetSum - set[currentIndex], currentSubset + [set[currentIndex]], currentIndex + 1)
    if withElement:
        return withElement
    return backtrack(set, targetSum, currentSubset, currentIndex + 1)

# Test cases 
print(subsetSum([3, 34, 4, 12, 5, 2], 9)) # [4, 5]
print(subsetSum([3, 34, 4, 12, 15, 2], 30)) # [3, 12, 15]
print(subsetSum([3, 34, 4, 12, 5, 2], 1)) # None
print(subsetSum([3, 34, 4, 12, 5, 2], 0)) # []
 