function createCourseCard(course) {
    const card = document.createElement('div');
    card.className = 'course-card';

    const title = document.createElement('h3');
    title.textContent = course.title;

    const description = document.createElement('p');
    description.textContent = course.description;

    const link = document.createElement('a');
    link.href = `course.html?id=${course.id}`;
    link.textContent = 'View Course';
    link.className = 'course-link';

    card.appendChild(title);
    card.appendChild(description);
    card.appendChild(link);

    return card;
}

export default createCourseCard;