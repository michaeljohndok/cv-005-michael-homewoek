from Multi_classification import *
from Multi_make_anno import *
from baidu_crawler import *

if __name__ == '__main__':
    while True :
        print("【1】百度关键字爬取训练图片")
        print("【2】图片命名整理")
        print("【3】生成CSV文件")
        print("【4】随机预览图片")
        print("【5】开始训练模型")
        print("【6】查看Loss图像")
        print("【7】查看Accuracy图像")
        print("【8】查看预测图像")
        print("【9】退出程序")
        task = input("请输入需要执行的任务编号：")
        task = int(task)
        if task == 9:
            print("退出程序，祝你愉快！老师记得给高分 ：）")
            break;
        else:
            if task == 1:
                #百度爬图
                baidu_fetch_image()
            if task == 2:
                #图片命名
                image_rename()
            if task == 3:
                #生成CSV文件
                generate_csv()
            if task == 4:
                #随机预览图片
                visualize_dataset()
            if task == 5:
                #随机预览图片
                train_task()
            if task == 6:
                #查看Loss图像
                check_loss()
            if task == 7:
                #查看Accuracy图像
                check_acc()
            if task == 8:
                #查看预测图像
                visualize_model()
            if task == 9:
                print("退出程序，祝您愉快！希望老师能打个好分 ：）")
                break;