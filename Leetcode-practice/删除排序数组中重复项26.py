# @TIME : 2019/4/9 上午7:41
# @File : 删除排序数组中重复项26.py
import time

class Solution:
    def removeDuplicates(self, nums) -> int:
        if not nums:
            return 0
        i = 0
        while i < len(nums) -1:
            if nums[i] == nums[i + 1]:
                del nums[i]
                continue
            else:
                i += 1

        return len(nums)


class Solution2:
    def removeDuplicates(self, nums) -> int:
        if not nums:
            return 0
        count = 0
        print("nums = ", nums)
        for i in range(len(nums)):
            if nums[count] != nums[i]:
                count += 1
                nums[count] = nums[i]
            print("nums = ", nums)
        return count + 1


start3 = time.time()

a = Solution()
nums = [1, 1, 2]
answer = a.removeDuplicates(nums)
end3 = time.time()
time3 = end3 - start3
print("answer = ", answer)
print("nums = ", nums)
print('程序1运行时间：', str(time3))


start2 = time.time()
b = Solution()
nums = [1, 1, 2]
answer = b.removeDuplicates(nums)
end2 = time.time()
time2 = end2 - start2
print("answer = ", answer)
print("nums = ", nums)
print('程序2运行时间：', str(time2))

# print(a.removeDuplicates([0,0,1,1,1,2,2,3,3,4]))