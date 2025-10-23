function generateLessonList(lessons, courseId) {
    const lessonListContainer = document.createElement('div');
    lessonListContainer.className = 'lesson-list';

    lessons.forEach((lesson, index) => {
        const lessonItem = document.createElement('div');
        lessonItem.className = 'lesson-item';

        const lessonTitle = document.createElement('h3');
        lessonTitle.textContent = lesson.title;

        const lessonProgress = document.createElement('input');
        lessonProgress.type = 'checkbox';
        lessonProgress.id = `lesson-${courseId}-${index}`;
        lessonProgress.checked = lesson.completed;
        lessonProgress.addEventListener('change', () => {
            lesson.completed = lessonProgress.checked;
            localStorage.setItem(`course-${courseId}`, JSON.stringify(lessons));
        });

        lessonItem.appendChild(lessonTitle);
        lessonItem.appendChild(lessonProgress);
        lessonListContainer.appendChild(lessonItem);
    });

    return lessonListContainer;
}

export default generateLessonList;