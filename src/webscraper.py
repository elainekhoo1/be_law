# create new conda env
# pip install selenium
import os
os.chdir(r'D:\00 Self-Learnings\0 Learning and Wellbeing (LAW)\be_law')
import pandas as pd
import numpy as np
from selenium import webdriver
from time import sleep
from datetime import timedelta, datetime

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.min_rows', 20)

# Create empty DataFrame to store the data.
columns = [
    'job_position',
    'company',
    'location',
    'career_level',
    'qualification',
    'years_of_exp',
    'job_type',
    'job_spec',
    'job_details',
    'salary',
    'posted_date'
]
jobs = pd.DataFrame(columns=columns)

def get_last_page(test_url):
    driver = webdriver.Chrome("./law2021_MVP/chromedriver")
    driver.get(test_url)
    list_pages=[x for x in driver.find_element_by_id('pagination').text]
    tabs_indices=[i for i, x in enumerate(list_pages) if x == "\n"]
    last_avai_page = set(range(tabs_indices[-1]+1, len(list_pages)))
    last_page=[list_pages[i] for i in last_avai_page]
    return int(''.join(last_page))

if __name__ == '__main__':

    job_number = -1
    test_url = 'https://www.jobstreet.com.my/en/job-search/data-scientist-jobs/'
    last_page=get_last_page(test_url)
    start_time = datetime.now()

    for page in range(1, last_page):
        page_url='https://www.jobstreet.com.my/en/job-search/data-scientist-jobs/{}/'.format(page)
        driver = webdriver.Chrome("./law2021_MVP/chromedriver")
        driver.get(page_url)
        sleep(8)
        # print(driver.find_element_by_xpath("//*[@id='jobList']/div[2]/div[1]/div[1]/div/span").text)
        for i in range(30):
            # each_job_path="//*[@id='jobList']/div[2]/div[3]/div/div[{}]/div/div/article/div/div/div[1]".format(i+1)
            try:
                each_job_path="//*[@id='jobList']/div[2]/div[3]/div/div[{}]/div/div/article/div/div/div[1]/div[1]/div[2]/h1/a".format(i+1)
                # each_job_path="//*[@id='jobList']/div[2]/div[3]/div/div[{}]/div/div/article/div/div/div[1]/div[1]".format(i+1)
                # WebDriverWait(driver, 10).until(expected_conditions.visibility_of_element_located((By.XPATH, each_job_path)))
                # WebDriverWait(driver, 20).until(expected_conditions.element_to_be_clickable((By.XPATH, each_job_path))).click()
                card = driver.find_element_by_xpath(each_job_path)
            except NoSuchElementException:
                each_job_path="//*[@id='jobList']/div[2]/div[3]/div/div[{}]/div/div/article/div/div/div[1]/div[1]/div[1]/h1/a".format(i+1)
                card = driver.find_element_by_xpath(each_job_path)

            driver.execute_script("arguments[0].click();", card)
            print("pass click")
            # driver.maximize_window()  # For maximizing window
            driver.implicitly_wait(20)  # gives an implicit wait for 20 seconds
            # sleep(20)

            common_path = "//*[@id='contentContainer']/div[2]/div/div/div[2]/div/div/div/div[2]/div/"

            job_position_path="div[1]/div/div[1]/div/div/div[1]/div/div/div[2]/div/div/div/div[1]/h1"
            job_position = driver.find_element_by_xpath(common_path+job_position_path).text
            driver.implicitly_wait(5)

            company_path="div[1]/div/div[1]/div/div/div[1]/div/div/div[2]/div/div/div/div[2]/span"
            company = driver.find_element_by_xpath(common_path+company_path).text
            driver.implicitly_wait(5)

            location_path = "div[1]/div/div[1]/div/div/div[2]/div/div/div/div[1]/div/span"
            try:
                location = driver.find_element_by_xpath(common_path+location_path).text
            except NoSuchElementException:
                location_path="div[1]/div/div[1]/div/div/div[2]/div/div/div/div[1]/div/div/div[2]/div/span"
                location = driver.find_element_by_xpath(common_path+location_path).text
            driver.implicitly_wait(5)

            try:
                driver.find_element_by_xpath(common_path + "div[2]/div/div[1]/div/div[3]/div/div[1]/h4")
                n=3
            except NoSuchElementException:
                n=2

            career_level_path = f"div[2]/div/div[1]/div/div[{n}]/div/div[2]/div/div/div[1]/div/div/div[2]/span"
            career_level = driver.find_element_by_xpath(common_path + career_level_path).text
            driver.implicitly_wait(5)

            qualification_path=f"div[2]/div/div[1]/div/div[{n}]/div/div[2]/div/div/div[2]/div/div/div[2]/span"
            qualification = driver.find_element_by_xpath(common_path+qualification_path).text
            driver.implicitly_wait(5)

            try:
                years_of_exp_path = f"div[2]/div/div[1]/div/div[{n}]/div/div[2]/div/div/div[3]/div/div/div[2]/span"
                years_of_exp = driver.find_element_by_xpath(common_path + years_of_exp_path).text
                job_type_path = f"div[2]/div/div[1]/div/div[{n}]/div/div[2]/div/div/div[4]/div/div/div[2]/span"
                job_type = driver.find_element_by_xpath(common_path + job_type_path).text
                job_spec_path = f"div[2]/div/div[1]/div/div[{n}]/div/div[2]/div/div/div[5]/div/div/div[2]/span"
                job_spec = driver.find_element_by_xpath(common_path + job_spec_path).text
            except NoSuchElementException:
                years_of_exp=np.nan
                job_type_path = f"div[2]/div/div[1]/div/div[{n}]/div/div[2]/div/div/div[3]/div/div/div[2]/span"
                job_type = driver.find_element_by_xpath(common_path + job_type_path).text
                job_spec_path = f"div[2]/div/div[1]/div/div[{n}]/div/div[2]/div/div/div[4]/div/div/div[2]/span"
                job_spec = driver.find_element_by_xpath(common_path + job_spec_path).text

            job_details_path = "div[2]/div/div[1]/div/div[1]/div/div[2]/div"
            try:
                job_details = driver.find_element_by_xpath(common_path + job_details_path).text
            except NoSuchElementException:
                job_details_path = "div[2]/div/div[1]/div/div[2]/div/div[2]/div/span/div"
                job_details = driver.find_element_by_xpath(common_path + job_details_path).text

            path_exist="div[1]/div/div[1]/div/div/div[2]/div/div/div/div[3]/span"
            posted_date_path="div[1]/div/div[1]/div/div/div[2]/div/div/div/div[2]/span"
            posted_date = driver.find_element_by_xpath(common_path+posted_date_path).text

            try:
                driver.find_element_by_xpath(common_path + path_exist)
                salary_path = "div[1]/div/div[1]/div/div/div[2]/div/div/div/div[2]/span"
                salary = driver.find_element_by_xpath(common_path + salary_path).text
                posted_date = driver.find_element_by_xpath(common_path + path_exist).text
            except NoSuchElementException:
                salary = np.nan
                pass

            job_number += 1
            # Save data in DataFrame.
            jobs.loc[job_number] = [
                job_position, company, location, career_level, qualification, years_of_exp,
                job_type, job_spec, job_details, salary, posted_date
            ]
            print(f"Pass Job {job_number}.")
        driver.quit()

    end_time = datetime.now()
    print(f"Looping through jobstreet jobs completed in {end_time - start_time}.")

    jobs.to_csv('law2021_MVP/data/data_scientist_jobstreet.csv', index=False)
    print("Job search save complete.")
