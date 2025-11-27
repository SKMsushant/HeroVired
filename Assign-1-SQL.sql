use lms;

-- 1. List All Courses with Their Category Names
-- Question: Retrieve a list of courses along with the name of the category to which each course belongs.
select course_name,category_name
from courses cr
left join categories ct
on cr.category_id=ct.category_id;

-- 2. Count the Number of Courses in Each Category
-- Question: For each category, count how many courses exist

select category_name,count(course_name) as course_count
from courses cr
left join categories ct
on cr.category_id=ct.category_id
group by 1;

-- 3. List All Students’ Full Names and Email Addresses
-- Question: Retrieve the full names and email addresses for all users with the role 'student'.

select concat(first_name," ",last_name) as full_name,email
from user
where role='student';

-- 4. Retrieve All Modules for a Specific Course Sorted by Module Order
-- Question: For a given course (e.g., course_id = 1), list its modules sorted by their order.

select m.course_id,course_name,module_name,module_order
from courses c
right join modules m
on c.course_id=m.course_id
where c.course_id=2
order by 4;

-- 5. List All Content Items for a Specific Module
-- Question: Retrieve all content items for a specific module (for example, module_id = 2).

select c.module_id,content_id,title,content_type,url
from modules m
right join content c
on m.module_id=c.module_id
where c.module_id=1;

-- 6. Find the Average Score for a Specific Assessment
-- Question: Calculate the average score of submissions for a given assessment (e.g., assessment_id = 1).

select assessment_id,round(avg(score),2) as avg_score
from assessment_submission
where assessment_id=5
group by assessment_id;


-- 7. List All Enrollments with Student Names and Course Names
-- Question: Retrieve a list of enrollments that shows the student’s full name, the course name, and the enrollment date.

select 
concat(u.first_name," ",u.last_name) as Fullname,
c.course_name,
e.enrolled_at
from enrollments e
join courses c
on e.course_id=c.course_id
join user u
on e.user_id=u.user_id;

-- 8. Retrieve All Instructors’ Full Names
-- Question: List the full names and email addresses of all users with the role 'instructor

select
concat(first_name,' ',last_name) as Full_name,
email as Email
from user
where role='instructor';

-- 9. Count the Number of Assessment Submissions per Assessment
-- Question: For each assessment, count how many submissions have been made.

select assessment_id,count(*) as submission_count
from assessment_submission
group by assessment_id;

-- 10. List the Top Scoring Submission for Each Assessment
-- Question: Retrieve, for each assessment, the submission that achieved the highest score

select assessment_id,max(score) max_score
from assessment_submission
group by assessment_id;

-- 11. Retrieve Courses Created After a Specific Date
-- Question: List courses that were created after '2023-04-01'.

select *
from courses
where created_at>'2023-04-01';

-- 12. Find Students Who Have Not Submitted Any Assessments
-- Question: Retrieve a list of students who do not have any records in the assessment_submission table.

select
u.user_id,
concat(u.first_name,' ',u.last_name) as Fullname,
u.email
from user u
left join assessment_submission e
on u.user_id=e.user_id
where e.user_id is null and role='student';

-- 13. List the Content for Courses in the 'Programming' Category
-- Question: Retrieve all content items for courses whose category is 'Programming'.

select *                            -- content_id,module_id,title,content_type,url
from content;
select *							-- category_id,category_name,description
from categories;
select *							-- course_id,course_name,description,category_id,created at
from courses;

select category_name,course_name,ct.description
from courses ct
join categories cg
on ct.category_id=cg.category_id
where category_name='Programming';
-- 14. Retrieve Modules That Have No Associated Content
-- Question: List modules that do not have any content items linked to them.

select 
m.module_id,
m.course_id,
m.module_name
from content c
left join modules m
on c.module_id=m.module_id
where c.module_id is null;

-- 15. List Courses with the Total Number of Enrollments
-- Question: For each course, display the course name along with the count of enrollments.


select c.course_id,c.course_name,count(e.enrollment_id) enrollment_count
from enrollments e
left join courses c
on e.course_id=c.course_id
group by 1
order by 3;

-- 16. Find the Average Assessment Submission Score for Each Course
-- Question: Calculate the average score of all assessment submissions for each course by joining courses,modules,assessments,and submissions.

select c.course_id,round(avg(atsm.score),2) as avg_score_per_course
from courses c
join modules m
on c.course_id=m.course_id
join assessments ats
on m.module_id=ats.module_id
join assessment_submission atsm
on ats.assessment_id=atsm.assessment_id
group by 1;

-- 17. List Users with Their Number of Enrollments
-- Question: Retrieve a list of all users along with the count of courses they are enrolled in.
select *                       -- user_id,first_name,last_name,email,password,role,created at
from user;
select *                       -- enrollment_id,course_id,user_id,enrolled_at
from enrollments;

select 
u.user_id,
concat(u.first_name,' ',u.last_name) as Full_name,
count(e.enrollment_id) as enrollment_count
from user u
left join enrollments e
on u.user_id=e.user_id
group by 1;

-- 18. Find the Assessment with the Highest Average Score
-- Question: Identify the assessment that has the highest average submission score

select ats.assessment_id,assessment_name,round(avg(atsm.score),2) as avg_score
from assessments ats
left join assessment_submission atsm
on ats.assessment_id=atsm.assessment_id
group by 1;

-- 19. List Courses Along with Their Modules and Content in Hierarchical Order
-- Question: Retrieve a hierarchical list that shows each course, its modules, and the content items within each module.

select
cr.course_id,
cr.course_name,
m.module_name,
m.module_order,
ct.title as module_contents
from courses cr
left join modules m
on cr.course_id=m.course_id
left join content ct
on m.module_id=ct.module_id
order by 1,4;

-- 20. Find the Total Number of Assessments Per Course
-- Question: For each course, count the total number of assessments available by joining courses, modules, and assessments

select
c.course_name,
count(a.assessment_id) count_assessments_per_course
from courses c
left join modules m
on c.course_id=m.course_id
left join assessments a
on m.module_id=a.assessment_id
group by 1;

-- 21. List All Enrollments from May 2023
-- Question: Retrieve all enrollment records where the enrollment date falls within May 2023.

select
concat(u.first_name,' ',u.last_name) as Full_name,
u.Email,
e.enrollment_id,e.course_id,
e.enrolled_at
from user u
left join enrollments e
on u.user_id=e.user_id
where e.enrolled_at > '2023-04-30 11:59:59' and e.enrolled_at < '2023-05-31 11:59:59';

-- 22. Retrieve Assessment Submission Details Along with Course and Student Information
-- Question: For each assessment submission, display the submission details along with the corresponding course name, student name, and assessment name

select
s.submission_id,
s.assessment_id,
s.user_id,
s.submitted_at,
s.score,
s.submission_data,
concat(u.first_name,' ',u.last_name) as Full_name,
c.course_name,
a.assessment_name,
a.assessment_type
from assessment_submission s
left join user u
on s.user_id=u.user_id
left join assessments a
on s.assessment_id=a.assessment_id
left join modules m
on a.module_id = m.module_id
left join courses c
on c.course_id=m.course_id;

-- 23. List All Users with Their Roles
-- Question: Retrieve a list of all users showing their full names and roles.
select
concat(first_name,' ',last_name) as Full_name,
role
from user;

-- 24. Find the Percentage of Passing Submissions for Each Assessment
-- Question: Assuming a passing score is 60 or above, calculate the passing percentage for each assessment.

select a.assessment_id,
a.assessment_name,
count(s.submission_id) as total_submissions,
sum(case when s.score>=60 then 1 else 0 end) as passing_submissions,
round(sum(case when s.score>=60 then 1 else 0 end)/count(s.submission_id),2) as passing_percentage
from assessments a
right join assessment_submission s
on a.assessment_id=s.assessment_id
where s.score>60
group by 1;

-- 25. Find Courses That Do Not Have Any Enrollments
-- Question: List the courses for which there are no enrollment records.

select *
from courses c
left join enrollments e
on c.course_id=e.course_id
where e.enrollment_id is null;
