import TeamCard from "../../components/TeamCard";

const team = [
  {
    name: "Chitkala H",
    role: "Project Lead",
    image: "/team/chitkala.jpg"
  },
  {
    name: "Amrutha",
    role: "Frontend Dev",
    image: "/team/Amrutha.jpg"
  },
  {
    name: "Deena Nishol dsouza",
    role: "Backend Dev",
    image: "/team/Deena.jpg"
  },
  {
    name: "Deviktha Hegde",
    role: "ML engineer",
    image: "/team/deviktha.jpg"
  }
];

export default function AboutPage() {
  return (
    <div>
      <h2 className="text-3xl font-semibold mb-8">Our Team</h2>
      <div className="grid md:grid-cols-4 gap-8">
        {team.map(member => (
          <TeamCard key={member.name} member={member} />
        ))}
      </div>
    </div>
  );
}
