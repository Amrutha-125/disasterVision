import Image from "next/image";

interface Member {
  name: string;
  role: string;
  image: string;
}

export default function TeamCard({ member }: { member: Member }) {
  return (
    <div className="bg-white rounded-xl shadow text-center p-6">
      <div className="w-32 h-32 mx-auto relative mb-4">
        <Image
          src={member.image}
          alt={member.name}
          fill
          sizes="128px"
          className="rounded-full object-cover"
        />
      </div>
      <h3 className="text-xl font-semibold">{member.name}</h3>
      <p className="text-gray-500">{member.role}</p>
    </div>
  );
}
