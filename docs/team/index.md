---
layout: page
title: The core osim-rl team
coreteam:
  - github: smsong
    name: Seungmoon Song
    task: Project leader
  - github: kidzik
    name: Łukasz Kidziński
    task: Project initiator & co-lead
  - github: carmichaelong
    name: Carmichael Ong
    task: OpenSim modeling
  - github: spMohanty
    name: Sharada P. Mohanty
    task: Baselines & crowdAI integration
  - github: chrisdembia
    name: Christopher Dembia
    task: OpenSim integration
  - name: Joy Ku
    task: Promotion
    image: https://cap.stanford.edu/profiles/viewImage?profileId=25840&type=square&ts=1509507382023
    link: http://nmbl.stanford.edu/people/joy-ku/
  - github: jenhicks
    name: Jen Hicks
    task: Biomechanics advisor
  - github: marcelsalathe
    name: Marcel Salathé
    task: Challenge advisor
  - name: Scott Delp
    task: Biomechanics advisor
    image: https://cap.stanford.edu/profiles/viewImage?profileId=6284&type=square&ts=1509499392349
    link: http://nmbl.stanford.edu/people/scott-delp/
  - name: Sergey Levine
    task: Reinforcement learning advisor
    image: https://pbs.twimg.com/profile_images/990434811662680064/BKCbJypl_400x400.jpg
    link: https://people.eecs.berkeley.edu/~svlevine/
  - github: opensim-org
    name: OpenSim core team
    task: OpenSim
contributors:
  - github: syllogismos
  - github: ctmakro
  - github: AdamStelmaszczyk
  - github: ViktorM
  - github: LiberiFatali
  - github: JackieTseng
  - github: seanfcarroll
  - github: gautam1858
  - github: seungjaeryanlee
---
<style>
.person {
text-align: center
}
.person img {
margin: 0.3em;
}
.person span {
display: block;
padding-top: 0.3em;
font-size: 0.8em;
}
</style>

{% for person in page.coreteam %}
{% assign loopindex = forloop.index0 | modulo: 4 %}
{% if loopindex == 0 or forloop.first %}
<div class="grid">
{% endif %}
<div class="unit one-fourth person">
{% if person.github %}
<a href="https://github.com/{{ person.github }}" class="post-author">
   {% avatar user=person.github size=150 %}<br />
   {{ person.name }}
</a>
{% else %}
<a href="{{ person.link }}" class="post-author">
<img src="{{ person.image }}" class="avatar" style="width: 150px;" /><br />
   {{ person.name }}
</a>
{% endif %}
<span>{{ person.task }}</span>
</div>
{% if loopindex == 3 or forloop.last %}
</div>
{% endif %}
{% endfor %}

<h3>Other notable contributors and users</h3>

{% for person in page.contributors %}
{% assign loopindex = forloop.index0 | modulo: 5 %}
{% if loopindex == 0 or forloop.first %}
<div class="grid">
{% endif %}
<div class="unit one-fifth person">
<a href="https://github.com/{{ person.github }}" class="post-author">
   {% avatar user=person.github size=120 %}<br />
   {{ person.github }}
</a>
</div>
{% if loopindex == 4 or forloop.last %}
</div>
{% endif %}
{% endfor %}


<div class="clear"></div>
