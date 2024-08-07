import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Blunder Mifflin", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)
st.write("""
### Sample Documents for RAGify

To showcase how RAGify works with Generative AI on custom documents, I created the Employee Handbook for a fictional company called Blunder Mifflin.

The chatbot answers questions related to the company policy.\n

So for example, if an employee of Blunder Mifflin wants to know what is the "Work from Home" policy, then they can just ask the chatbot that question and get the answer using the power of Generative AI.

If the company's Work From Home policy gets updated, they just update the documents, no changes needed to the chatbot. The chatbot starts providing answers based on latest information.

### Input PDF files

I fed the Company Policy of "Blunder Mifflin" to RAGify in the form of 5 PDF documents:

1. Blunder Mifflin's History and Mission
2. Employee Handbook
3. In-Office Relationships Policy
4. Prank Protocol
5. Birthday Party Committee Rules

You can now chat with these documents using the Chatbot accessible from the sidebar.

In the following sections, you'll find the contents of the 5 PDF files (the Company Policy).

---

### About Blunder Mifflin

#### Our History

Founded in 1785 by the visionary Michael Blunder, Blunder Mifflin started in Cranton, a town famous for... well, not much. Built on principles like quality, reliability, and a stubborn refusal to modernize, we've grown from a tiny local supplier to a name recognized by at least a few people nationwide.

#### Our Mission

At Blunder Mifflin, we aim to empower businesses with top-notch office supplies and customer service that makes you wonder if we're serious. We know the right tools are crucial for productivity, so we offer a vast array of office supplies to meet every conceivable need, and then some.

#### Our Products

Blunder Mifflin's product range is extensive, almost overwhelmingly so:

- **Stationery** - Pens, pencils, notebooks - basically everything you needed in elementary school.
- **Office Equipment** - Printers, scanners, and shredders for when you want to destroy evidence.
- **Furniture** - Ergonomic chairs and desks, because comfort is key when avoiding work.
- **Breakroom Supplies** - Coffee makers and snacks to keep you fueled and slightly less cranky.
- **Cleaning Supplies** - Disinfectants and trash bags, because cleanliness is next to impossible.
- **Technology** - Computers and networking gear to keep you connected and endlessly frustrated.

#### Our Team

- **Regional Manager** - The fearless leader whose motivational speeches are as frequent as they are unnecessary. Leadership through chaos is the mantra.
- **Assistant (to the) Regional Manager** - A beet-loving enforcer with martial arts skills, always on high alert for office security threats that never materialize.
- **Sales Representative** - Master of closing deals and orchestrating pranks. Balances charm with just enough mischief to keep things interesting.
- **Receptionist** - The artistic soul of the office, greeting visitors with a smile and dreaming of a more creative future.
- **Accounts Department** - The no-nonsense team ensuring our finances are precise. They run a tight ship and occasionally crack a smile for pets and snacks.
- **Warehouse Department** - The muscle of Blunder Mifflin, moving products efficiently while keeping the office staff humble.

#### Join Us on Our Journey

Whether you're a small business owner, a corporate executive, or just someone in need of office supplies, we're here to serve you.

Thank you for choosing Blunder Mifflin. Together, we can turn any workplace into a semi-functional environment.

### Blunder Mifflin Employee Handbook

#### As a Blunder Mifflin Employee

Welcome to Blunder Mifflin, where you're our most valuable resource (after the coffee machine). We're committed to being a top institution in Cranton. Here's what we expect from you:

- Support the company's mission (whatever it is this week).
- Know your job well (Google is your friend).
- Promote inclusivity (treat everyone nicely).
- Do your job accurately and professionally (at least look like you're trying).
- Own your actions (blaming the intern is so last year).
- Be punctual (or have a good excuse).
- Maintain high ethics (no office gossip...unless it's really good).
- Understand your workplace role (no, you can't be Regional Manager).
- Be a team player (unless you're working alone).
- Communicate clearly (use spellcheck).
- Listen and respond (nod and smile).
- Show trust and respect (even if it's hard).
- Pursue personal growth (read a book occasionally).
- Know the policies (skim them once).

This handbook is your guide to surviving...oops, we mean thriving at Blunder Mifflin.

#### Remote and Hybrid Work Policy

Blunder Mifflin values in-person collaboration (we love seeing your face), but remote and hybrid work are options if they fit the job.

- **Fully at Office** - You're here all the time. I mean, you have no personal life anywhich way.
- **Hybrid** - Split your time between the office and your couch.
- **Fully Remote** - Work from anywhere, but show up occasionally.

#### Staff Grievance Procedure

Got a problem? Here's how to solve it:

1. Talk to your supervisor.
2. Talk to your supervisor's supervisor.
3. Contact Employee Relations.
4. Climb the management ladder.
5. Final appeal to the President (good luck).

#### Drug, Alcohol, and Smoking Policy

Blunder Mifflin promotes wellness (cue laughter). No drugs or alcohol at The Office. If you need help, we have resources.

Smoking and vaping are banned everywhere, including a 50-foot perimeter around all facilities. Violators will face disciplinary action. Complaints go to HR or Security, depending on who's breaking the rules.

Compliance is mandatory. No negotiation. Stay healthy, folks.

### In-Office Relationships Policy

#### In-Office Relationships

Blunder Mifflin recognizes that love is in the air (or maybe it's just the printer toner), but office romances can complicate things. Here's our take:

1. **Disclosure** - If you find yourself smitten with a colleague, disclose the relationship to HR. We love a good office gossip, but let's keep it official.
2. **Professionalism** - Keep your relationship professional during office hours. Save the PDA for after 5 PM (or at least behind closed doors).
3. **No Favouritism** - Romantic involvement shouldn't lead to favouritism. Promotions should be based on merit, not on how many dinners you've shared.
4. **Avoid Conflicts** - Relationships between supervisors and their direct reports are strongly discouraged. If it happens, one of you might need a new boss (or a new job).

#### Nepotism

At Blunder Mifflin, family is everything - until it affects the workplace. Here's the lowdown on nepotism:

1. **Hiring** - If you want to hire your cousin, sister, or that uncle who needs a job, disclose it to HR first. We have enough family drama without adding yours.
2. **No Preferential Treatment** - Family members working together should not receive preferential treatment. Your relative still needs to meet deadlines and attend meetings like everyone else.
3. **Chain of Command** - Family members should not report directly to one another.
4. **Transparency** - Be transparent about any familial relationships in the workplace.

### Prank Protocol

At Blunder Mifflin, we believe that a little humour can brighten the workday. However, to keep the office environment friendly and professional, we've established the following guidelines for pranks:

1. **Respect Personal Space and Property** - No pranks that invade personal space or damage property. Encasing a colleague's stapler in jelly is funny; rearranging their desk in the bathroom is not.
2. **Safety First** - Pranks should never endanger anyone's safety. No setting up tripwires or hiding hazardous materials in unusual places.
3. **Productivity Matters** - Pranks should not interfere with work. Timing is key - save the elaborate schemes for break times or after hours.
4. **Inclusivity** - Pranks should be inclusive and not target specific individuals repeatedly. Everyone loves a good laugh, but let's keep it fair.
5. **Good Taste** - Keep it tasteful. Avoid pranks that could be considered offensive or discriminatory. Humour should bring us together, not push us apart.
6. **Clean-Up** - The prankster is responsible for cleaning up after their pranks. This includes removing all traces of jelly, wrapping paper, or whatever medium was used.

### Birthday Party Committee Rules

At Blunder Mifflin, we take birthday celebrations seriously. To ensure that everyone's special day is recognized without causing chaos, we've established the following rules:

1. **Party Planning Committee (PPC)** - The PPC is responsible for organizing all birthday celebrations.
2. **Budget** - Each birthday celebration has a budget of 20 bucks. This includes decorations, cake, and any miscellaneous expenses. Spend wisely; we're not made of money.
3. **Cake Flavours** - A variety of cake flavours will be rotated to accommodate different tastes. No more endless debates over chocolate vs. vanilla. Special dietary needs should be communicated in advance.
4. **Timing** - Birthday parties will be held in the break room at 3 PM on the Friday closest to the birthday. This way, we maximise attendance and minimise disruption.
5. **Decorations** - Keep decorations simple and office-appropriate. Balloons, streamers, and a banner are fine; life-sized cardboard cutouts of the birthday person are not.
6. **Participation** - Everyone is encouraged to participate. If you're not into singing “Happy Birthday,” a polite clap will suffice.
7. **Gifts** - Office-wide gifts are not required but are welcome. If you choose to give a gift, keep it appropriate for the workplace.
8. **Clean-Up** - The Party Planning Committee is responsible for setting up and cleaning up after the party. However, everyone should pitch in to keep the break room clean.
9. **Complaints** - Any complaints about birthday parties should be directed to the Party Planning Committee. We'll try to address them, but remember, you can't please everyone.
10. **Surprise Parties** - Surprise parties are permitted but must be coordinated with the PPC to avoid scheduling conflicts and ensure the birthday person doesn't get surprised out of the office.

By following these rules, we can ensure that every birthday at Blunder Mifflin is a memorable (in a good way) occasion.
				 """)
